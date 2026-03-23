class Config:
    image_size = 256
    patch_size = 16         
    num_patches = image_size // patch_size

    in_channels = 3
    feature_dim = 256      

    memory_size = 128       
    top_k = 5             
    temperature = 1.0      

    transformer_dim = 256
    num_heads = 4
    dropout = 0.1

    mask_prob = 0.5        


    # GENERATORS dim
    hidden_dim = 256

    disc_channels = 64

    lambda_t = 1.0
    lambda_s = 1.0
    lambda_dist = 0.1
    lambda_gen = 0.01
    lambda_dis = 0.01

    lr = 1e-4
    batch_size = 16
    num_epochs = 100

    use_normalization = True   
