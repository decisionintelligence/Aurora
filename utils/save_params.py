from aurora.modeling_aurora import AuroraForPrediction

model = AuroraForPrediction.from_pretrained("/home/Aurora/checkpoints/Aurora_Release_Version")

print(model.num_parameters())