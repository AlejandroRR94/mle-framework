establecer_remote:
# Establecemos el bucket de s3 al que haremos push
	dvc remote add -d mys3remote s3://model-experiment --force
	git commit .dvc/config -m "Iniciando DVC remote"

add_data:
# Añadimos la carpeta cuyos datos queremos versionar
	dvc add data

push_data:
# Hacemos push a los datos
	dvc push