#		src

src directory (source) is where the source code lies. 

##		Direcotories

###		Setup

General setup of the subsystem directories
```
src/
├── subsystem/
│   ├── source_files.py
│   ├── README.md
│   └── tests/
```
Each subsytem is in it's own folder with their source files and tests directory. The tests directory holds the pytests for the source file (the unit tests). Add a readme to each directory to explain usage and descriptions. 

###		Current Dir Tree

```
RobosubCV/
├── assets/
│   ├── images/
│   └── pdf/
├── src/
│   ├── augmentation/
│   │   ├── geometric/
│   │   ├── photometric/
│   │   └── tests/
│   ├── roboflow_datasets/
│   │   └── tests/
│   └── training/
│   	└── tests/
└── sbatch/
```

(Explination of the different directories)
