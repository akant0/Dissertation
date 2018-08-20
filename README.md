Sample of code for my dissertation on ultrasound images in audio-visual automatic speech recognition for children with speech disorders.
Ultrasound database not available, as it is owned by the University of Edinburgh.

"CNN.py" was created as initial experimentation with the ultrasound images. 
It is a convolutional neural net for image classification. 
Phone target labels can be clustered by place of articulation.

"autoencoder" is a convolutional autoencoder created to extract a 128 dimensional feature vector for the ultrasound images. Experiments were run on training different data quantities and epoch numbers, and using those features in the Kaldi AV-ASR system.

"AEreconstruction" saves a sample image reconstructed by the autoencoder, and the original to compare. 


