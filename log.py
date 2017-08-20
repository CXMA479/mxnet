
class mytick(object):
	def __init__(self,proc='proc',logging=None):
		self.cur=0
		self.s_time =None# time.time()
		self.eplapse =0
		self.proc=proc
		self.logging=logging
	def load(self):
		self.s_time=time.time()
		logging.info('starting the process, please wait for the progress bar...')
		
	def tick(self,idx,num):
		cur_tmp = int(idx*100/num)
		if cur_tmp > self.cur:
			self.cur = cur_tmp
			self.elapse = round(time.time() - self.s_time)
			self.s_time = time.time()
			logging.info('%s:\t %d%s elapsed:\t%d sec'%(self.proc,self.cur,'% ,',int(self.elapse)))
