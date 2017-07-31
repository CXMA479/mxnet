"""
There is no intersection between the returned two symbols except the lower's outputs and the higher's inputs (they are same).
So, if you concatenate the models guided by the two symbols the final ouputs are same as the input symbol's.
Usually, this function is to handle feature extraction and feature retrieving.


Chen Y. Liang

"""


import mxnet as mx
import numpy as np
import json




def track_op(nodes,op,node_lst,depth=100,keep_pattern=''):
	"""
                     {u'inputs': [[0, 0, 0]], u'attr': {u'scalar': u'1'}, u'name': u'_plusscalar3', u'op': u'_plus_scalar'}
	"""

#	if op['op'] == 'null':# data is preserved cause they could be weight or bais...
#		print(op['op'])		
#		return
	if depth<=0:
#		print('reach the maximum depth')
		return
	inputs = op['inputs']
	input_num = len(inputs)
	for input_node in inputs:
		input_idx = input_node[0]
		if (nodes[input_idx]['op'] == 'null') or (keep_pattern in nodes[input_idx]['op']) and len(keep_pattern) >0:
			return
		node_lst.append(input_idx)
		new_op = nodes[input_idx]
		track_op(nodes,new_op,node_lst,depth-1)
	

def splitbyname(sym,opname,keep_pattern='',idx_base=1):
	"""
                  the returned symbol will take opname as the 1st layer
		inspired by mx.viz.plot_network
	"""
	conf = json.loads(sym.tojson())
	nodes = conf['nodes']
	arg_nodes = conf['arg_nodes']
	heads = conf['heads']
	node_row_ptr = conf['node_row_ptr']
	attrs      = conf['attrs']
	pred_node_lst =[]  # record input nodes to be sweeped out
	par_node_lst=[]
	old_port_names=[]
	new_port_names=[]
	new_nodes=json.loads(sym.tojson())['nodes'] # do it again
	for op in new_nodes:
		"""
                      {u'inputs': [[0, 0, 0]], u'attr': {u'scalar': u'1'}, u'name': u'_plusscalar3', u'op': u'_plus_scalar'}
		"""
		if op['name'] in opname:
			"""
                                 get the input, set its input node to be the null, record its predecessors
			"""

			track_op(new_nodes,op,par_node_lst,1,keep_pattern) # just parents for final input ports
#			track_op(nodes,op,pred_node_lst)

		"""
                        modify the op whose name doesnot contain string of weight or bias
		"""
	par_node_lst=list(np.unique(par_node_lst))
	for idx,par_idx in enumerate(par_node_lst):
		old_port_names += [new_nodes[par_idx]['name']]
		new_port_names += ['feat_%d'%(idx+idx_base)]

		new_nodes[par_idx]['name'] = 'feat_%d'%(idx+idx_base)
		nodes[par_idx]['name'] = 'feat_%d'%(idx+idx_base)
		new_nodes[par_idx]['op']='null'
		new_nodes[par_idx]['inputs'] =[]
		if 'attrs' in new_nodes[par_idx]:
			new_nodes[par_idx].pop('attrs')
	new_arg_nodes = par_node_lst
	"""
	for pred_idx in pred_node_lst:
		if pred_idx in par_nodes_lst:
			continue
	"""
	top_conf={'nodes':new_nodes,'arg_nodes':new_arg_nodes,'heads':heads,'node_row_ptr':node_row_ptr,'attrs':attrs}
	top_symbol = mx.sym.load_json(json.dumps(top_conf))

	
	new_heads=[]
	for arg_idx in par_node_lst:
		new_heads += [[arg_idx,0,0]]

	bottom_conf={'nodes':nodes,'arg_nodes':arg_nodes,'heads':new_heads,'node_row_ptr':node_row_ptr,'attrs':attrs}
	bottom_symbol = mx.sym.load_json(json.dumps(bottom_conf))

  
	return bottom_symbol,top_symbol, old_port_names, new_port_names
