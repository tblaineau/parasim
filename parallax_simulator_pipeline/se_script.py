import argparse
import os
import numpy as np
from parallax_simulator_pipeline.similarity_estimator import count_peaks, compute_distances

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', '-n', type=str, required=True, help="output file name")
	parser.add_argument('--parameter_file', type=str, required=True,
						help="template name of parameter file, ex : parameters1M_")
	parser.add_argument('--nb_line_pmsfile', type=int, required=True,
						help="Number of light curve per parameters file.")
	parser.add_argument('--nb_files', type=int, required=True, help="Number of parameters file.")
	parser.add_argument('--nb_jobs', type=int, required=False, default=100, help="Number of jobs.")
	parser.add_argument('--current_job', type=int, required=True, help="current job number")

	args = parser.parse_args()

	nb_line_pmsfile = args.nb_line_pmsfile
	nb_files = args.nb_files
	nb_jobs = args.nb_jobs
	current_job = args.current_job

	nb_samples_job = int(nb_line_pmsfile * nb_files / nb_jobs)
	factor = int(nb_line_pmsfile / nb_samples_job)

	print(nb_line_pmsfile, nb_files, nb_jobs, nb_samples_job)

	assert nb_line_pmsfile > 0, 'Invalid number of line per file.'
	assert current_job > 0, 'Invalid current job number.'
	assert nb_samples_job - (nb_line_pmsfile * nb_files / nb_jobs) == 0, f'nb_samples_job is not an integer : {nb_line_pmsfile * nb_files / nb_jobs}'
	assert int(nb_line_pmsfile / nb_samples_job) - (nb_line_pmsfile / nb_samples_job) == 0, f'factor is not an integer :{nb_line_pmsfile / nb_samples_job}'

	params_file_idx = (current_job - 1) // factor
	params_line_idx = (current_job - 1) % factor

	start = params_line_idx * nb_samples_job
	end = (params_line_idx + 1) * nb_samples_job

	output_name = args.name
	parameter_file = args.parameter_file + '_' + str(params_file_idx) + '.npy'

	assert os.path.isfile(parameter_file), 'Parameter file does not exist : ' + parameter_file

	compute_distances(output_name, distance=count_peaks, parameter_list=np.load(parameter_file, allow_pickle=True),
					  start=start, end=end)
