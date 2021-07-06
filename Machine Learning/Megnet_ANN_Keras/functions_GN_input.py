import numpy as np

def make_input(file_list):
	atom_fea_list = []
	bond_fea_list = []
	glob_fea_list = []
	bond_idx_atom1_list = []
	bond_idx_atom2_list = []
	atom_idx_list = []
	bond_idx_list = []
	E_list = []
	Ga_idx_list = []
	As_idx_list = []

	num_lattice = 9
	stacked_atom_len = 0
	for f in range(0, len(file_list)):
		infile2 = open(file_list[f][:-1],'r')
		data_list = infile2.readlines()
		infile2.close()

##### define the basic information of structure #####
		cell = data_list[4:7]
		atoms = data_list[num_lattice:]

		cell_list = []
		for i in range(len(cell)):
		        cell_list.append(cell[i].split()[0:3])
		cell_matrix = np.asfarray(cell_list)

		atom_list = []
		for i in range(len(atoms)):
			atom_list.append(atoms[i].split()[1:4])
		atom_matrix = np.asfarray(atom_list)

##### define the feature and index of bonds #####
		if f == 0:
			for i in range(len(atoms)):
				for j in range(len(atoms)):
					d_min = sum((atom_matrix[i,:] - atom_matrix[j,:])**2)**(1/2)
					for l1 in [-1,0,1]:
						for l2 in [-1,0,1]:
							for l3 in [-1,0,1]:
								d = sum((atom_matrix[i,:] - atom_matrix[j,:] + l1*cell_matrix[0,:] + l2*cell_matrix[1,:] + l3*cell_matrix[2,:])**2)**(1/2)
								d_min = min([d_min, d])
					if d_min <= 3.0 and i != j:
						bond_fea_list.append([1.0/d_min])
						bond_idx_atom1_list.append(i+len(atom_idx_list))
						bond_idx_atom2_list.append(j+len(atom_idx_list))
						bond_idx_list.append(f)
			num_edge = len(bond_idx_list)
		else:
			for i in range(num_edge):
				d_min = sum((atom_matrix[bond_idx_atom1_list[i],:] - atom_matrix[bond_idx_atom2_list[i],:])**2)**(1/2)
				for l1 in [-1,0,1]:
					for l2 in [-1,0,1]:
						for l3 in [-1,0,1]:
							d = sum((atom_matrix[bond_idx_atom1_list[i],:] - atom_matrix[bond_idx_atom2_list[i],:] + l1*cell_matrix[0,:] + l2*cell_matrix[1,:] + l3*cell_matrix[2,:])**2)**(1/2)
							d_min = min([d_min, d])
				bond_fea_list.append([1.0/d_min])
				bond_idx_atom1_list.append(bond_idx_atom1_list[i]+len(atom_idx_list))
				bond_idx_atom2_list.append(bond_idx_atom2_list[i]+len(atom_idx_list))
				bond_idx_list.append(f)

##### define the feature and index of atoms #####
		for i in range(len(atoms)):
			atom_idx_list.append(f)
			if 'Ga' in atoms[i].split():
				atom_fea_list.append([0.,0.,1.,0.,0.,0.,0.,0.])
				Ga_idx_list.append(stacked_atom_len + i)
			elif 'As' in atoms[i].split():
				atom_fea_list.append([0.,0.,0.,0.,1.,0.,0.,0.])
				As_idx_list.append(stacked_atom_len + i)

####### define the global feature and E #######
		glob_fea_list.append(np.empty([0]))
		E_list.append([float(data_list[0].split()[4])])
		stacked_atom_len = stacked_atom_len + len(atoms)

	Ga_grp_list = list(np.array(atom_idx_list)[Ga_idx_list])
	As_grp_list = list(np.array(atom_idx_list)[As_idx_list])
	return [atom_fea_list, bond_fea_list, glob_fea_list, bond_idx_atom1_list, bond_idx_atom2_list, atom_idx_list, bond_idx_list, Ga_idx_list, As_idx_list, Ga_grp_list, As_grp_list], E_list

def standarization(E_list):
	data_E_matrix = np.asfarray(E_list)
	num_data = data_E_matrix.shape[0] # num of data
	data_E_mean = data_E_matrix.mean()
	data_E_std = 0
	for i in range(num_data):
		data_E_std += (data_E_matrix[i]-data_E_mean)**2

	data_E_std = (data_E_std/(num_data-1))**0.5
	data_stE_matrix = (data_E_matrix-data_E_mean)/data_E_std # standarized target E

	return list(data_stE_matrix), data_E_mean, data_E_std

def concat_bulk_surf(bulk_inputs, bulk_E, surf_inputs, surf_E, batch_size, E_mean, E_std):
	surf_inputs[3] = list(np.array(surf_inputs[3]) + len(bulk_inputs[5]))
	surf_inputs[4] = list(np.array(surf_inputs[4]) + len(bulk_inputs[5]))
	surf_inputs[5] = list(np.array(surf_inputs[5]) + batch_size)
	surf_inputs[6] = list(np.array(surf_inputs[6]) + batch_size)
	surf_inputs[7] = list(np.array(surf_inputs[7]) + len(bulk_inputs[5]))
	surf_inputs[8] = list(np.array(surf_inputs[8]) + len(bulk_inputs[5]))
	surf_inputs[9] = list(np.array(surf_inputs[9]) + batch_size)
	surf_inputs[10] = list(np.array(surf_inputs[10]) + batch_size)

	atom_fea = np.asfarray(bulk_inputs[0] + surf_inputs[0])
	atom_fea = np.expand_dims(atom_fea, axis=0)
	bond_fea = np.asfarray(bulk_inputs[1] + surf_inputs[1])
	bond_fea = np.expand_dims(bond_fea, axis=0)
	glob_fea = np.asfarray(bulk_inputs[2] + surf_inputs[2])
	glob_fea = np.expand_dims(glob_fea, axis=0)
	bond_idx_atom1 = np.asarray(bulk_inputs[3] + surf_inputs[3])
	bond_idx_atom1 = np.expand_dims(bond_idx_atom1, axis=0)
	bond_idx_atom2 = np.asarray(bulk_inputs[4] + surf_inputs[4])
	bond_idx_atom2 = np.expand_dims(bond_idx_atom2, axis=0)
	atom_idx = np.asarray(bulk_inputs[5] + surf_inputs[5])
	atom_idx = np.expand_dims(atom_idx, axis=0)
	bond_idx = np.asarray(bulk_inputs[6] + surf_inputs[6])
	bond_idx = np.expand_dims(bond_idx, axis=0)
	Ga_idx = np.asarray(bulk_inputs[7] + surf_inputs[7])
	Ga_idx = np.expand_dims(Ga_idx, axis=0)
	As_idx = np.asarray(bulk_inputs[8] + surf_inputs[8])
	As_idx = np.expand_dims(As_idx, axis=0)
	Ga_grp = np.asarray(bulk_inputs[9] + surf_inputs[9])
	Ga_grp = np.expand_dims(Ga_grp, axis=0)
	As_grp = np.asarray(bulk_inputs[10] + surf_inputs[10])
	As_grp = np.expand_dims(As_grp, axis=0)

	E = bulk_E + surf_E
	stand_E = (np.asfarray(E)-E_mean)/E_std
	stand_E = np.expand_dims(np.asfarray(stand_E), axis=0)

	X = [atom_fea, bond_fea, glob_fea, bond_idx_atom1, bond_idx_atom2, atom_idx, bond_idx, Ga_idx, As_idx, Ga_grp, As_grp]
	#X = [[atom_fea], [bond_fea], [glob_fea], [bond_idx_atom1], [bond_idx_atom2], [atom_idx], [bond_idx], [Ga_idx], [As_idx], [Ga_grp], [As_grp]]
	return X, stand_E
