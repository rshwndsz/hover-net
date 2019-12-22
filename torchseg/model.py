import segmentation_models_pytorch as smp

# Model pretrained on imagenet
# See: https://github.com/qubvel/segmentation_models.pytorch/
model = smp.Unet(encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm=True,
                 decoder_channels=[256, 128, 64, 32, 16],
                 # See: https://arxiv.org/pdf/1808.08127.pdf
                 decoder_attention_type=None,
                 activation=None,
                 in_channels=3,
                 classes=1
                 )
