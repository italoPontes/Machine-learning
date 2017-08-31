#########################################################
#Federal University of Campina Grande (UFCG)		#
#Machine Learning Course				#
#PhD student: Ítalo de Pontes Oliveira			#
#e-mail: italooliveira at copin dot ufcg dot edu dot br	#
#Professor: Leandro Balby Marinho			#
#########################################################

#Installing all Dependencies

#Update System
sudo apt -y update

#Install Numpy and other dependencies
#source: https://www.scipy.org/install.html
sudo apt -y install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

#Install Scikit-learn
#source: http://scikit-learn.org/stable/install.html
sudo pip install -U scikit-learn

#Install Anaconda 4.4.0
#source: https://www.tensorflow.org/install/install_linux#InstallingAnaconda
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
chmod +x Anaconda2-4.4.0-Linux-x86_64.sh
./Anaconda2-4.4.0-Linux-x86_64.sh

#Installing Tensorflow with Anaconda
conda create -n tensorflow
source activate tensorflow
# CPU Only
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl

#Install docker
sudo apt -y install docker
docker run -it gcr.io/tensorflow/tensorflow bash

#Test Tensorflow without GPU
python test_tensorflow.py
