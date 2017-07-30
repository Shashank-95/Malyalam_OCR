import cv2
import numpy as np
import os
import collections
import unicodedata
import csv
from PIL import Image
from collections import defaultdict
class OCR:
    glyphs=[0,1,2,3,4,6,7,8,11,12,13,14,15,16,18,19,21,23,24,25,26,27,28,29,31,32,33,35,36,37,38,39,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,59,61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,79,81,82,83,84,85,86,88,89,91,92,101,102]
    def __init__(self,fname):
        """"
        """
        orgpath="/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam"
        os.chdir(orgpath)
        img=Image.open(fname+'.tif')
        img.save(fname+'.png')
        img=cv2.imread(fname+'.png')
        self.height,self.width = img.shape[:2]
        self.img=img
        binimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        binimg=cv2.medianBlur(binimg,5)
        self.binimg=binimg
        invimg=cv2.bitwise_not(binimg)
        self.invimg=invimg
        
        
        
    def wordSegmentation(self,ppimg,sterlh,sterlw,fname):
        """
        page to word segment
        :param ppimg:
        :param sterlh:
        :param sterlw:
        :param fname:
        :return:
        """
        kernel=np.ones((sterlh,sterlw),dtype="uint8")*255
        dilation=cv2.dilate(ppimg,kernel,iterations=4)
        #cv2.imwrite('dilated.png',dilation)
        orgimg=cv2.bitwise_not(dilation)
        output=cv2.connectedComponentsWithStats(dilation,8)
        self.num_words=output[0]
        labels=output[1]
        stats=output[2]
        centroids=output[3]
        dirname='OCR_of_'+fname
        os.mkdir(dirname)
        os.chdir(dirname)
        cv2.imwrite('dilated_image.png',dilation)
        wname='words_of_'+fname
        os.mkdir(wname)

        #print self.num_words
        self.worddict={}
        #word_positions=open('Bounding_Boxes.txt','a')
        for i in range(1,len(stats)):
            y=stats[i,cv2.CC_STAT_TOP]
            w=stats[i,cv2.CC_STAT_WIDTH]
            x=stats[i,cv2.CC_STAT_LEFT]
            h=stats[i,cv2.CC_STAT_HEIGHT]
            self.worddict[i]=[x,y,w,h]
            #word_positions.write(str(i)+'\t'+'=>'+'\t \t '+str(self.worddict[i]) +'\n')
            crop=self.img[y:y+h,x:x+w]
            
            cv2.imwrite(os.path.join(wname,str(i)+'.png'),crop)
        #print self.worddict

        
        cv2.imwrite('contours.png',self.img)
        cv2.imwrite('binary.png',ppimg)
        orgpath="/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam"
        os.chdir(orgpath)

    def charSegmentation(self,fname,num):
        dirname='OCR_of_'+fname
        os.chdir(dirname)
        chname='characters_of_'+fname
        os.mkdir(chname)
        self.ttlchars=0
        self.chardict=defaultdict(list)
        for s in range(1,num):
            lst1=[]
            back='/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam/OCR_of_'+fname+'/words_of_'+fname
            os.chdir(back)
            word = cv2.imread(str(s)+'.png',0)
            invimg=cv2.bitwise_not(word)
            thresh_color = cv2.cvtColor(word,cv2.COLOR_GRAY2BGR)
            output=cv2.connectedComponentsWithStats(invimg,4)
            num_chars=output[0]
            labels=output[1]
            stats=output[2]
            centroids=output[3]
            
            self.ttlchars = self.ttlchars + num_chars
            folder='/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam/OCR_of_'+fname+ '/characters_of_'+ fname
            os.chdir(folder)
            newch='charsOf'+str(s)
            os.mkdir(newch)
            for k in range(1,len(stats)):
                y=stats[k,cv2.CC_STAT_TOP]
                w=stats[k,cv2.CC_STAT_WIDTH]
                x=stats[k,cv2.CC_STAT_LEFT]
                h=stats[k,cv2.CC_STAT_HEIGHT]
                
                self.chardict[s].append(x)
                mask=np.ones(word.shape[:2],dtype="uint8")*255
                
                if w*h > 200:
                    for i in range(y,y+h):
                        for j in range(x,x+w):
                            if labels[i][j]==k:
                                mask[i][j]=0
                    crop=mask[y:(y+h),x:(x+w)]
                    resize_char=cv2.resize(crop,(100,100))
                    cv2.imwrite(os.path.join(newch,str(k)+'.png'),resize_char)
                
            
                
            cv2.imwrite(os.path.join(newch,'word.png'),thresh_color)              
        #print self.chardict   
        orgpath="/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam"
        os.chdir(orgpath)
        print str(self.num_words) +':'+ str(self.ttlchars)

    @staticmethod
    def generateSkinBone():
        for i in OCR.glyphs:
            dirname='/home/shashank/Desktop/Data set for training/'+ str(i)
            os.chdir(dirname)
            andimg=np.ones((100,100),dtype='uint8')*255
            orimg=np.ones((100,100),dtype='uint8')*0
            p=1
            while(cv2.imread(str(p*10)+'.png',0) is not None):
                img=cv2.imread(str(p*10)+'.png',0)
                andimg=cv2.bitwise_and(img,andimg)
                orimg=cv2.bitwise_or(img,orimg)
                p+=1

            cv2.imwrite('skin.png _'+ str(i) ,andimg)
            cv2.imwrite('bone.png _'+ str(i),orimg)
            fbname='/home/shashank/Desktop/Data set for training/feature_extraction_data/bones'
            os.chdir(fbname)
            cv2.imwrite('bone_'+ str(i)+'.png' ,orimg)
            fsname='/home/shashank/Desktop/Data set for training/feature_extraction_data/skins'
            os.chdir(fsname)
            cv2.imwrite('skin_'+ str(i)+'.png' ,andimg)
            orgpath="/home/shashank/Desktop/Geeksfolder/imageprocfiles"
            os.chdir(orgpath)

            
    def feature_extraction(self,fname,num_words):
        orgpath="/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam/OCR_of_"+fname
        dirname='OCR_of_'+fname
        os.chdir(dirname)

        
        word_positions=open('Bounding_Boxes.txt','a')
        djvutxt=open(fname+'.txt','a')
        djvutxt.write('(page 0 0 '+str(int(0.75*self.width))+'\t'+str(int(0.75*self.height))+'\n')

        
        feature_folder='recognised_characters'
        os.mkdir(feature_folder)
        os.chdir(feature_folder)
        for x in range(1,self.num_words):
            
            templst=[]
            tempdict={}
            charname='rec_charsOf'+str(x)
            os.mkdir(charname)
            os.chdir(orgpath)
            chfolder='characters_of_'+fname
            os.chdir(chfolder)
            inner='charsOf'+str(x)
            os.chdir(inner)
            a=1
            while(cv2.imread(str(a)+'.png',0) is not None):
                mainchr=cv2.imread(str(a)+'.png',0)
                
                skprob={}
                bnprob={}
                data='/home/shashank/Desktop/Data set for training/feature_extraction_data'
                os.chdir(data)
                for k in OCR.glyphs:
                    cnt=0
                    ttl=0
                    skin=cv2.imread('skins/skin_'+ str(k) + '.png',0)
                    matchs=cv2.matchTemplate(mainchr,skin,cv2.TM_CCOEFF)
                    skprob[matchs[0][0]]=k
            
                self.odskprob=collections.OrderedDict(sorted(skprob.items()))


                for p in OCR.glyphs:
                    cntb=0
                    ttlb=0
                    bone=cv2.imread('bones/bone_'+ str(p) + '.png',0)
                    matchb=cv2.matchTemplate(mainchr,bone,cv2.TM_CCOEFF)
                    bnprob[matchb[0][0]]=p
                        
                self.odbnprob=collections.OrderedDict(sorted(bnprob.items()))

                pros=max(self.odskprob.keys())
                prob=max(self.odbnprob.keys())
                recskin=skprob[max(self.odskprob.keys(),key=float)]
                recbone=bnprob[max(self.odbnprob.keys(),key=float)]
                if recskin != recbone:
                    if pros>prob:
                        finalrec=recskin
                    else:
                        finalrec=recbone
                else:
                    finalrec=recskin

                
                templst=self.chardict[x]
                
                tempdict[templst[a-1]]=finalrec
                
                
                    
                    
                     
                
                os.chdir(orgpath)
                recdir='recognised_characters'
                os.chdir(recdir)
                charname='rec_charsOf'+str(x)
                os.chdir(charname)
                skbnfdr=str(a)+'_recprobs'
                os.mkdir(skbnfdr)
                cv2.imwrite(os.path.join(skbnfdr,'rec_final_'+str(finalrec)+'.png'),mainchr)
                #cv2.imwrite(os.path.join(skbnfdr,'prob_bone_rec_'+str(recbone)+'.png'),mainchr)
                
                     
                os.chdir(orgpath)
                inner1='characters_of_'+ fname +'/charsOf'+str(x)
                os.chdir(inner1)
                a=a+1
            just=[]    
            tempdict=collections.OrderedDict(sorted(tempdict.items()))
            for i,(key,value) in enumerate(tempdict.iteritems()):
                just.append(value)
                
            csvpath='/home/shashank/Desktop/Geeksfolder/imageprocfiles/Malayalam'
            os.chdir(csvpath)
            with open('mapping_datamod.csv','rb') as csvfile:
                unicode_dict={}
                read=csv.DictReader(csvfile,('glyph_num','unicode'))
                for row in read:
                    unicode_dict[row['glyph_num']]=row['unicode']

            unicode_list=[]
            g=0
            skip=False
            for z in just:
                if z==88 or z==89:
                    unicode_list.append(unicode_dict[str(just[g+1])])
                    unicode_list.append(unicode_dict[str(just[g])])
                    skip=True
                    g+=2
                    continue
                if skip == True:
                    skip=False
                    continue   
                    
                unicode_list.append(unicode_dict[str(z)])

            unicode_string=''.join(unicode_list)
            #unicode_string=''.join(unicode_list)
            #print unicode_string
            #final_string=open('final_string','a')
            #final_string.write(unicode_string+'\n')

            
            
            #print templst
            #print tempdict
            sf=0.75
            templist=self.worddict[x]
            x1=templist[0]
            y1=templist[1]
            w1=templist[2]
            h1=templist[3]


            xmin=int(x1*sf)
            ymin=int((self.height-y1-h1)*sf)
            xmax=int((x1+w1)*sf)
            ymax=int((self.height-y1)*sf)
                
                
            djvutxt.write('\t(word '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)  +' \"'+unicode_string+ '\")\n')
            word_positions.write(str(x)+'=>'+str(self.worddict[x]) + '=> FinalRec:'+unicode_string+'\n')
            del just[:]
            del templst[:]
            tempdict.clear()

        
    
            os.chdir(orgpath)
            backfdr='recognised_characters'
            os.chdir(backfdr)

        djvutxt.write(')')


        

    
                            
                    
                    
                
        
        
OCR.generateSkinBone()

for i in range(0,48):
##    if i>0 and i<10:
##        numstr='000'+str(i)
##    elif i>=10 and i<100:
##        numstr='00'+str(i)
##    elif i>100:
    numstr=str(i)
#numstr=str(0)
    image7=OCR(numstr)
    image7.wordSegmentation(image7.invimg,3,8,numstr+'.png')
    image7.charSegmentation(numstr+'.png',image7.num_words)
    image7.feature_extraction(numstr+'.png',image7.num_words)
    
        




cv2.waitKey(0)
cv2.destroyAllWindows()

