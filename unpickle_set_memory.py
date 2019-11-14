def grow_tf_memory() -> None:
    import tensorflow as tf

    if tf.__version__ < "2.0.0":
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
    else:
        # It looks like Keras now automatically picks up the underlying TF
        # configuration
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True
    return None


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pickle", "unpickle", "load"))
    args = parser.parse_args()

    if args.action == "pickle":
        pass
    elif args.action == "unpickle":
        pass
    elif args.action == "load":
        grow_tf_memory()
        from keras.models import load_model

        model = load_model("issue_3676.model")
        model.summary()
    else:
        pass
