# EOS
Equation of state (EOS) script for fitting energy-volume data obtained from the calculation in the full-potential [ELK](https://github.com/nskorikov/exciting-plus) code, with the account of the DMFT correction to total energy received in the framework of the DMFT calculation in the  [AMULET](http://amulet-code.org/) code.

The input file for fitting the EOS consists of columns:
1. lattice parameter (just for convenience, it is not used in calculations)
2. the volume of the unit cell in a.e ^ 3
3. DFT total energy, Hartree
4. DMFT correction to total energy, eV
5. Relative error for DMFT energy correction, eV

There can be any number of column but read are only the five first column.
All values should be collected directly from DFT and DMFT output files without any conversion of units. The script will convert energy to eV and volume to \AA^3.
You can see command line options of script using -h key.
