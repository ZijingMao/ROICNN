def inference_pooling_L2norm_filter(images, kwidth=5):
    # channel domain pooling mapper
    kheight=2 
    split_dim = 1   # 1 represents split on spatial domain
    input_image_list = split_eeg.split_eeg_signal_axes(images,
                                                       split_dim=split_dim)
    input_image_length = len(input_image_list)

    # the pooling mapper should choose half size of the image size
    pool_s, _ = concat_eeg.pool_eeg_signal_channel(input_image_list, input_image_length/2, 1)
    _print_tensor_size(pool_s)

    pool_s = tf.mul(pool_s,pool_s)
    pool_s = tf.mul(float(kwidth), pool_s)

    pool_s = tf.nn.avg_pool(pool_s, ksize=[1, 1, kwidth, 1],
                            strides=[1, 1, kwidth, 1], padding='VALID')

    pool_s = tf.sqrt(pool_s)

    pool_s = tf.nn.max_pool(pool_s, ksize=[1, kheight, 1, 1],
                             strides=[1, kheight, 1, 1], padding='VALID')

    _print_tensor_size(pool_s)

    return pool_s
