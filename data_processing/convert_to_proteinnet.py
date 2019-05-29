#!/usr/bin/python

import numpy as np
import sys

if __name__ == '__main__':
	stem = sys.argv[1]

	i_fa = open(stem, 'r')
	name = i_fa.readline()[1:]
	seq  = "".join([line.strip() for line in i_fa.readlines()]) + '\n'
	header = '[ID]\n' + name + '[PRIMARY]\n' + seq + '[EVOLUTIONARY]'
	i_fa.close()

	i_icinfo = open(stem + '.icinfo', 'r')
	i_cinfo  = open(stem + '.cinfo', 'r')
	evos = []
	for buf_icinfo in range(9): buf_icinfo = i_icinfo.readline()
	for buf_cinfo in range(10): buf_cinfo  = i_cinfo.readline()

	while buf_icinfo != '//\n':
		buf_icinfo_split = buf_icinfo.split()
		if buf_icinfo_split[0] != '-':
			ps = np.array([float(p) for p in buf_cinfo.split()[1:]])
			ps = ps / np.sum(ps)
			evo = np.append(ps, float(buf_icinfo_split[3]) / np.log2(20))
			evos.append(np.tile(evo, 2))
		buf_icinfo = i_icinfo.readline()
		buf_cinfo  = i_cinfo.readline()

	i_icinfo.close()
	i_cinfo.close()

	np.savetxt(stem + '.proteinnet', np.stack(evos).T, fmt='%1.5f', comments='', header=header)
	with open(stem + '.proteinnet', 'a') as o: o.write('\n')
