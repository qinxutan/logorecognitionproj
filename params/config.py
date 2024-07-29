
class Config:
    """ Exec parameters """
    ## Dataset dirs
    training_dir = "/Users/qinxutan/Documents/htxinternship/logorecognition/logos_full/train/"
    testing_dir = "//Users/qinxutan/Documents/htxinternship/logorecognition/logos_full/test/"

    # Betternet 224,224 following pytorch doc
    im_w = 224
    im_h = 224

    ## Model params
    model = "betternet"

    pretrained = True
    distanceLayer = True  # defines if the last layer uses a distance metric or a neuron output
    bceLoss = False  # If true uses Binary cross entropy. Else: contrastive loss

    train_batch_size = 3
    train_number_epochs = 30
    lrate = 0.05

    #Weights of Each Feature
    ocr_weight = 0.4
    orb_weight = 0.3
    nh_weight = 0.3
    combined_weight = 0.8
    siamese_weight = 0.2

    #Threshold for Each Feature
    ocr_threshold = 0.1
    orb_threshold = 0.1
    nh_threshold = 0.1

    #combined similarity threshold in testing
    best_match = 0

    ## Model save/load paths
    best_model_path = "testmodel"
    model_path = "testmodel_last"