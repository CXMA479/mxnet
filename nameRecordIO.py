"""
encoding  image names is trival and always drives me crazy which also decays my passion.

  Chen Y. Liang
  Aug 20, 2017
"""
import mxnet as mx
import cPickle as cpk
import os, logging, time
import numpy as np
class  nameRecordIO(object):
    """
    1. get your data directly by its name!
        rec.write_name(name,data)
        data=rec.read_name(name)

    2. appending operation is supported!
        nameRecordIO(idx_path, uri, 'a' )
    """
    def __init__(self, name , flag, key_type=int ):
        idx_path = name+'.idx'
        uri      = name+'.rec'
        self.file_path= name
        self.cpk_path=self.file_path+'.cpk'
        self.data = None # for storing pre-loaded data if mm
        self.flag = flag

        if os.path.isfile(self.cpk_path):  # load it
            f=open(self.cpk_path,'r')
            try:
                self.nameDict = cpk.load(f)
            except: # if empty, len(rec.idx.keys() ) == 0
                self.nameDict = {}

            assert  type(self.nameDict) == dict,'%s not a dict!'%self.cpk_path
            f.close()
          # how many has been record...
            self.count = len(self.nameDict.keys())
        else:
          # make a new file, dict...
            self.count = 0
            self.nameDict = {}

        self.idx_path = idx_path
        self.uri      = uri

        if flag is 'a': # create a new one
            rec_src = mx.recordio.MXIndexedRecordIO(idx_path,uri,'r',key_type)
        # check consistency...
            assert len(rec_src.idx.keys()) == len(self.nameDict.keys()),\
               'exptected equal, got %d vs %d'%(len(rec_src.idx.keys()) , len(self.nameDict.keys()))
      # make a new one
            self.rec = mx.recordio.MXIndexedRecordIO(idx_path+'.tmp',uri+'.tmp','w',key_type)

            print('copy to a new record, please wait...')
            for idx in rec_src.idx.keys():
                data = rec_src.read_idx(idx)
                #print data
                self.rec.write_idx(idx,data)
            print('ok.')
            rec_src.close()
        # delete the src, rename the new one
            os.remove(idx_path)
            os.remove(uri)

        else:# as usual...
            self.rec = mx.recordio.MXIndexedRecordIO(idx_path,uri,flag,key_type)
            if flag == 'w':
                self.nameDict = {}
                self.count = 0
        logging.info('There are %d record in the file[%s].'%\
                    (len(self.nameDict),uri))
        self.flag = flag
        if len(self.nameDict):
            logging.info('samples of key: %s'%(', '.join(self.nameDict.keys()[:10]) ) )

    def loadAllData(self):
        """
            load all data into memory
        """
        assert self.flag=='r',self.flag
        logging.info('load all data into memory...')
        data = {k:self.read_name(k) for k in self.nameDict}
        self.data = data
        logging.info('done.')

    def IObenchmark(self, decode_fn=None):
        """
            on random access
        """
        loop = 10
        time_acc = 0.
        print('Starting IObenchmark...')
        name_list = self.nameDict.keys()
        sample_num = len(name_list)
        for i in xrange(loop):
            print('Loop:\t%d/%d...'%(i,loop))
            np.random.shuffle(name_list)
            t1 = time.time()
            for name in name_list:
                decode_fn(self.read_name(name)) if decode_fn is not None \
                    else self.read_name(name)
            time_acc += time.time() - t1
        self.sample_speed = loop*sample_num/time_acc
        print('Benchmark:\tdecode:%s\t%dloop(s) cost %f sec\tSpeed: %f samples/sec'%('True'if decode_fn else 'False', loop, time_acc, self.sample_speed) )



    def write_name(self, name, buf):
     # allocate a ID then, call write_idx
        if name in self.nameDict: # override
            self.rec.write_idx(self.nameDict[name],buf)
        else:
            self.nameDict[name] = self.count
            self.rec.write_idx(self.count, buf)
            self.count += 1
 
    def read_name(self,name):
        assert name in self.nameDict,'%s does not exist in Dict!'%name#, self.nameDict[10])
        return self.rec.read_idx(self.nameDict[name]) if \
                self.data is None  else self.data[name]
 
    def close(self):
     # dunmp the dict
        import cPickle as cpk
        if self.flag is not  'r': # overwrite only writing or appendding
            with open(self.cpk_path,'w') as f:
                cpk.dump(self.nameDict,f)
        self.rec.close()
        if os.path.isfile(self.idx_path+'.tmp'): # rename the new one
            os.rename(self.idx_path+'.tmp',self.idx_path)
            os.rename(self.uri+'.tmp', self.uri)

    def __del__(self):
        self.close()


if __name__  == '__main__':
  rec=nameRecordIO('test','w')
  d1='chen, yos'
  d2='lisdg, 20xxxx71, xxx08xx09t'
  imgname1='img1'
  imgname2='img2'
  rec.write_name(imgname1,d1)
  rec.write_name(imgname2,d2)
  rec.close()
  
  rec=nameRecordIO('test','r')
  assert rec.read_name(imgname1)==d1
  assert rec.read_name(imgname2) == d2
  rec.close()
  
  
  rec=nameRecordIO('test','a')
  d3='chen, '
  d4=', 20xsds20xxx9t'
  imgname3='img3'
  imgname4='img4'
  rec.write_name(imgname3,d3)
  rec.write_name(imgname4,d4)
  rec.close()
  
  rec=nameRecordIO('test','r')
  rec.loadAllData()
  #print rec.nameDict
  assert rec.read_name(imgname1) == d1, rec.read_name(imgname1)
  assert rec.read_name(imgname2) == d2, rec.read_name(imgname2)
  assert rec.read_name(imgname3) == d3, rec.read_name(imgname3)
  assert rec.read_name(imgname4) == d4
  
  rec.close()
  print 'test ok!'


