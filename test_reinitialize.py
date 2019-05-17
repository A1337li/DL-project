from network import*

my_model, conv_part = create_model([512, 256], 3)
my_model, my_conv_model, input_size = reinitialize_final_layers(my_model, conv_part, 11)
print()
print("Full model")
for layer in my_model.layers:
	print(layer.name)
print()

conv_model = get_conv_part(my_model)

print("Conv part")
for layer in conv_model.layers:
	print(layer.name)

print_layer_trainable(conv_model)