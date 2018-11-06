def copyModelToModel(model_source, model_target, certain_layer=""):        
	for target_layer, sourse_layer in zip(target_model.layers, source_model.layers):
		weights = sourse_layer.get_weights()
		target_layer.set_weights(weights)
		if target_layer.name == certain_layer:
			break
	print("model source was copied into model target")


def copy_model_weights(model_source, model_target, start_layer=249):
	num_layers = len(model.layers)
	print(len(source_model.layers), 'layers in source model')
	print(num_layers, 'layers in target model')
	for i in range(start_layer, num_layers):
		weights = source_model.layers[i].get_weights()
		model.layers[i].set_weights(weights)
		model.layers[i].trainable = True

