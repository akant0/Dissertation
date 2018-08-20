# 1. Converts feature matrices (one matrix per utterance, of dimension N FRAMES X N FEATURES)
# to the same shape as the audio matrices (an offset bc of the ultrasound
# tech which is slightly delayed compared to audio, the amount changing per utt)
# 2. removes speech therapist speech from feature matrices (bc the recording
# therapy sessions have therapist and child speech mixed)
# Speech therapist vs child speech labels obtained from the CSTR at the Uni
# of Edinburgh
# Features in this example: 100 dimensional discrete cosine transform
# Data: disordered UXSSD
import os
import scipy
import scipy.io as sio
import numpy as np
x='train'
save=True
remove_slt=True
subdirs=os.listdir('/kaldi/egs/s2/data/data_uxssd/')
for subset in subdirs: # i.e. therapy groups: Mid, Post, etc.
    # Kaldi-made file with utt and number of frames:
    audframes='/kaldi/egs/s2/data/data_uxssd/'+subset+'/utt2num_frames'
    audF={}
    with open(audframes, 'r') as f:
        for L in f:
            L=L.strip().split()
            utt=L[0]
            count=L[1]
            audF[utt]=count
    largerVpath='/DCT/UXSSD-new/'
    mats=os.listdir(largerVpath)  # matrices to convert
    savepath='/DCT/padded_feats/UXSSD/'+subset+'/'
    completed=os.listdir(savepath)
    for m in sorted(mats):
        if m.endswith('mat'):
            F=largerVpath+m
            utt=m[:-4]
            array=scipy.io.loadmat(F) # load the feat. matrix
            matrix=array["ults_matrix"]
            matrix=np.transpose(matrix) # if matrices are not framesXfeatures
            if utt in audF:
                if m not in completed: # if matrix not yet converted
                    print(utt)
                    # print(matrix)
                    #print(audF[utt], 'aud frames')
                    #print(matrix.shape, 'vis frames')
                    diff = int(audF[utt])-int(matrix.shape[0])
                    if diff < 0:
                        new=np.delete(matrix, slice(0, -diff), axis=0)
                    else:
                        emp=np.zeros((diff, 100))
                        new=np.concatenate((emp,matrix ), axis=0)
                    #save new mat
                    if save==True:
                        scipy.io.savemat(savepath+m, mdict={utt: new})
if remove_slt==True:
    for subset in subdirs:
        if subset in subdirs:
            print(subset)
            UXTD_SPK_LABS='/data/labels/UXTD/speaker-labs/lab'
            UXSSD_SPK_LABS='/data/labels/UXSSD/speaker-labs/lab'
            UPX_SPK_LABS='/data/labels/UPX/speaker-labs/lab'
            rSLTpath='/DCT/padded_remSLT/UXSSD/'+subset+'/'
            savepath='/DCT/padded_feats/UXSSD/'+subset+'/'
            padded=os.listdir(savepath)
            labs=os.listdir(UXSSD_SPK_LABS)
            vismats=padded
            lab2=[]
            visfeats2=[]
            for i in labs:
                lab2.append(i.replace('.lab','' ))
            for i in vismats:
                visfeats2.append(i.replace('.mat','' ))
            todo=[]
            for i in visfeats2:
                if i in lab2:
                    todo.append(i+'.lab')
            for m in sorted(todo):
                F=savepath+m
                utt=m[:-4]
                array=scipy.io.loadmat(F.replace('.lab', '.mat'))
                matrix=array[utt]
                before=matrix.shape[0]
                total=0
                if m in os.listdir(UXSSD_SPK_LABS):
                    with open(os.path.join(UXSSD_SPK_LABS, m)) as f:
                        for L in f:
                            L=L.strip().split()
                            if L[2]=='SLT': # if the speech is therapist
                                start=int(L[0])
                                end=int(L[1])
                                start_frame = int(float(start)/100000.)
                                end_frame   = int(float(end)/100000.)
                                dur=end_frame-start_frame
                                dim=100 # 100d DCT features
                                total+=dur
                                # replace the SLT speech with zeros
                                repl = np.zeros((dur, dim))
                                if matrix.shape[0]<end_frame:
                                    print('BAD', m) # then something is wrong
                                else:
                                    matrix[start_frame:end_frame, :] = repl.copy()
                                with open('rem-SLT-changes', 'a') as x:
                                    x.write(utt+' ')
                                    x.write('before: ')
                                    x.write(str(before))
                                    x.write(' After: ')
                                    x.write(str(matrix.shape[0]))
                                    x.write(' Diff: ')
                                    x.write(str(dur)+ ' ')
                                    x.write('Cumulative: ')
                                    x.write(str(total))
                                    x.write('\n')
                    scipy.io.savemat(rSLTpath+m.replace('.lab', '.mat'), mdict={utt: matrix})
