from network import*

my_model, conv_part = create_model([512, 256], 3)
my_model, my_conv_model = reinitialize_final_layers(my_model, conv_part, 11)
print()
print("Full model")
for layer in my_model.layers:
	print(layer.name)
print()
print("Conv part")
for layer in my_conv_model.layers:
	print(layer.name)

print_layer_trainable(my_model)