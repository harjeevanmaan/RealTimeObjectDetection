import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

def load_weights(model,cfgfile,weightfile):
    fp = open(weightfile, "rb")
    np.fromfile(fp, dtype=np.int32, count=5)
    blocks = parse_cfg(cfgfile)

    for i, block in enumerate(blocks[1:]):

        if (block["type"] == "convolutional"):
            conv_layer = model.get_layer('conv_' + str(i))
            print("layer: ",i+1,conv_layer)

            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            
            if "batch_normalize" in block:
                norm_layer = model.get_layer('bnorm_' + str(i))
                print("layer: ",i+1,norm_layer)

                size = np.prod(norm_layer.get_weights()[0].shape)
                bn_weights = np.fromfile(fp, dtype=np.float32, count=4*filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, "Failed to read all data in file"
    fp.close()

def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as fp:
        lines = [line.rstrip('\n') for line in fp if (line != '\n' and line[0] != '#')]

    holder = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            line = "type=" + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split('=')
        holder[key.rstrip()] = value.lstrip()

    blocks.append(holder)
    return blocks
    
def yolov3(cfgfile, model_size, num_classes):   #this project took heavy inspiration from this blog post
                                                #https://mc.ai/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/
    blocks = parse_cfg(cfgfile)
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    inputs = input_image = tf.keras.layers.Input(shape=model_size)
    inputs = inputs / 255.0

    for i, block in enumerate(blocks[1:]):

        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            if strides > 1:
                inputs = tf.keras.layers.ZeroPadding2D(padding = ((1, 0), (1, 0)))(inputs)

            inputs = tf.keras.layers.Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding= ("valid" if strides > 1 else "same"),
                            name=("conv_" + str(i)),
                            use_bias=(False if ("batch_normalize" in block) else True))(inputs)

            if "batch_normalize" in block:
                inputs = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i))(inputs)
                inputs = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            inputs = tf.keras.layers.UpSampling2D(size=stride)(inputs)

        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])

            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)

            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]

        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)
            out_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [-1, n_anchors*out_shape[1]*out_shape[2], 5 + num_classes])
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)
            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])
            strides = (input_image.shape[1]//out_shape[1], input_image.shape[2]//out_shape[2])
            box_centers = (box_centers + cxy) * strides
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1
        outputs[i] = inputs
        output_filters.append(filters)

    model = tf.keras.Model(input_image, out_pred)
    model.summary()
    
    return model

def main():
    model_path = "/home/jeevan/code/tf/yolov3/yolov3.h5"
    class_file = "/home/jeevan/code/tf/yolov3/coco.names"
    weights_file = "/home/jeevan/code/tf/yolov3/yolov3.weights"
    config_file = "/home/jeevan/code/tf/yolov3/yolov3.cfg"

    model_size = (416, 416, 3)
    num_classes = 80

    model = yolov3(config_file, model_size, num_classes)
    load_weights(model, config_file, weights_file)
    
    model.save(model_path)

if __name__ == "__main__":
    main()