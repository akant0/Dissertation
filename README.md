Folder "AV-ASR":

Sample of code for my dissertation on ultrasound images in audio-visual automatic speech recognition for children with speech disorders.
Ultrasound database not available, as it is owned by the University of Edinburgh.

1. "CNN.py" was created as initial experimentation with the ultrasound images. 
It is a convolutional neural net for image classification. 
Phone target labels can be clustered by place of articulation.

2. "autoencoder.py" is a convolutional autoencoder created to extract a 128 dimensional feature vector for the ultrasound images. Experiments were run on training different data quantities and epoch numbers, and using those features in the Kaldi AV-ASR system.

3. "AEreconstruction.py" saves a sample image reconstructed by the autoencoder, and the original to compare. 

4. "fix_US_matrices.py" alters the ultrasound feature matrices by 1. making them the same number of frames as audio data (as there is a delay in the US technology when recording), 2. removes speech therapist's speech feature frames (fills with zeros) because the recordings have both speech pathologist and child speech.

5. "run_DNN_AV-ASR.sh" is a sample of the Kaldi AV-ASR feature fusion at the DNN-level. 

Folder "Synthesis":

1. "synth.py" is a simple concatenative speech synthesizer. 
Requires a folder "monophones" with wav files for each monophone (not uploaded, as it belongs to the University of Edinburgh)
2. Example1: -p "HELLO. i was born {22/01} with {3.14} or, 344 cats"
3. Example2: -p "A rose by any other name would smell as sweet"

** Still to upload: Festival speech synthesis projects **
