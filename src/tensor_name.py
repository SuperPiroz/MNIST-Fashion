import tensorflow as tf

# Ange sökvägen till din embedding-katalog
ckpt_path = 'outputs/logs/embedding'

# Hitta senaste checkpoint
ckpt = tf.train.latest_checkpoint(ckpt_path)
print("Checkpoint path:", ckpt)

# Läs ut alla variabelnamn i checkpointen
reader = tf.train.load_checkpoint(ckpt)
print("Variabelnamn i checkpointen:")
for name in reader.get_variable_to_shape_map().keys():
    print(name)