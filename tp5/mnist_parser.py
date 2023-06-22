archivo_entrada = "tp5/MNIST_train.txt"
archivo_salida = "tp5/MNIST_parsed_not_normalized.txt"

with open(archivo_entrada, 'r') as file:
    lineas = file.readlines()

with open(archivo_salida, 'a') as file:
    for linea in lineas:
        numeros = linea.strip('()\n').split(',')

        for i, numero in enumerate(numeros, 1):
            file.write(numero.strip())
            if i % 28 == 0:
                file.write('\n')
            else:
                file.write(' ')