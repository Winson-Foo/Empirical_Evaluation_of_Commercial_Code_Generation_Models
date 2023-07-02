import logging

class VOLPY(object):
    def __init__(self, template_size=0.02, context_size=35, censor_size=12, 
                 visualize_ROI=False, flip_signal=True, hp_freq_pb=1/3, nPC_bg=8, ridge_bg=0.01,  
                 hp_freq=1, clip=100, threshold_method='adaptive_threshold', min_spikes=10, 
                 pnorm=0.5, threshold=3, sigmas=np.array([1, 1.5, 2]), n_iter=2, weight_update='ridge', 
                 do_plot=False, do_cross_val=False, sub_freq=20, 
                 method='spikepursuit', superfactor=10, params=None):
        if params is None:
            logging.warning("Parameters are not set from volparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params

        self.estimates = {}

    def fit(self):
        results = []
        fnames = self.params.data['fnames']
        fr = self.params.data['fr']
        method = self.params.volspike['method']
        volspike = spikepursuit.volspike if method == 'spikepursuit' else atm.volspike
        N = len(self.params.data['index'])
        
        for j in range((N + n_processes - 1) // n_processes):
            li = [k for k in range(j * n_processes, min((j + 1) * n_processes, N))]
            args_in = []
            
            for i in li:
                idx = self.params.data['index'][i]
                ROIs = self.params.data['ROIs'][idx]
                weights = self.params.data['weights'][i] if self.params.data['weights'] else None
                args_in.append([fnames, fr, idx, ROIs, weights, self.params.volspike])

            if dview:
                results_part = dview.map_sync(volspike, args_in)
            else:
                results_part = list(map(volspike, args_in))
            
            results.extend(results_part)
        
        for i in results[0].keys():
            try:
                self.estimates[i] = np.array([results[j][i] for j in range(N)])
            except:
                self.estimates[i] = [results[j][i] for j in range(N)]
                
        return self