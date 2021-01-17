import segmentation_models as sm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def makeModel(optimizer='Adam', encoder = 'resnet18', encoder_freeze=False):
    preprocess_input = sm.get_preprocessing(encoder)

    model = sm.Unet(encoder, classes=1, encoder_weights='imagenet', encoder_freeze=encoder_freeze)
    model.compile(
        optimizer=optimizer,
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    return model, preprocess_input