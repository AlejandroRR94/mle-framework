push_data: add_data
# Hacemos push a los datos
	dvc push
	echo "DATA PUSHED TO S3 SUCCESSFULLY!"

add_data: establish_remote
# Añadimos la carpeta cuyos datos queremos versionar
	dvc add data

establish_remote:
# Establecemos el bucket de s3 al que haremos push
	dvc remote add -d mys3remote s3://model-experiment --force
# git commit .dvc/config -m "Iniciando DVC remote"