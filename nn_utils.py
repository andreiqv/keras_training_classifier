def copyModelToModel(source_model, target_model, certain_layer=""):        
	for target_layer, sourse_layer in zip(target_model.layers, source_model.layers):
		weights = sourse_layer.get_weights()
		target_layer.set_weights(weights)
		if target_layer.name == certain_layer:
			break
	print("model source was copied into model target")


def copy_model_weights(source_model, target_model, start_layer=249):
	num_layers = len(target_model.layers)
	print(len(source_model.layers), 'layers in source model')
	print(num_layers, 'layers in target model')
	for i in range(start_layer, num_layers):
		weights = source_model.layers[i].get_weights()
		target_model.layers[i].set_weights(weights)
		target_model.layers[i].trainable = True


def copy_top_weights_to_model(source_top_model, target_model, start_layer=249):
	num_layers = len(target_model.layers)
	print(len(source_top_model.layers), 'layers in source model')
	print(num_layers, 'layers in target model')

	for i_source_0, i_target in enumerate(range(start_layer, num_layers)):
		i_source = i_source_0 + 1
		source_layer = source_top_model.layers[i_source]
		target_layer = target_model.layers[i_target]
		weights = source_layer.get_weights()
		target_layer.set_weights(weights)
		print('{}->{}; weights={}, shape={}; {} -> {}'.format(i_source, i_target, len(weights), source_layer.input_shape, source_layer.name, target_layer.name))
		#print('source layer:', source_layer.name, ', input_shape =', source_layer.input_shape)
		#print('target layer:', target_layer.name, ', input_shape =', target_layer.input_shape)
		#target_layer.trainable = True


def reset_weights(model, start_layer=249):
	layers = model.layers
	session = K.get_session()
	for i in range(start_layer, len(layers)):
		if hasattr(layers[i], 'kernel_initializer'):
			layers[i].kernel.initializer.run(session=session)
