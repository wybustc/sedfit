import numpy as num  
import matplotlib.pyplot as plt  

class binlc:
    def __init__(self,mjd,flux, ferr,method='median',node_method='gap', gap=1,Nsep=5,sep=None ,show=True, Nmin=2, bin1=True):
        #here we use the flux as an representative, and in fact magitude is also accepted 
        ind=num.argsort(mjd) 
        self.mjd=mjd[ind]
        self.flux=flux[ind]
        self.ferr=ferr[ind]
        self.show=show
        if bin1==True:
            self.binning(method=method,node_method=node_method,gap=gap,Nsep=Nsep, sep=sep, Nmin=Nmin) 
        


    def prod_nodes(self,node_method='gap', gap=1 ,Nsep=5,sep=None, nodes_low=None, nodes_up=None  ): 
        #There are two methods 'gap' and 'nearby'
        #gap is the paramter for method of gap 
        #Nsep and sep are parameters about 'nearby', In the nearby method, those point with separation < Nsep*sep was binned together 
        if (nodes_low is not None) & (nodes_up is not None):
            if not len(nodes_low)==len(nodes_up): raise Exception('nodes_low and nodes_up should have same length ')
            self.nodes_low=num.array(nodes_low) 
            self.nodes_up =num.array(nodes_up ) 
        elif node_method=='gap': 
        #    print('produce the nodes with gap=%d'%gap) 
            nodes=[self.mjd[0]]
            node = self.mjd[0]
            for mjd in self.mjd:
                if mjd<=node+gap:
                    continue 
                else: 
                    node=mjd
                    nodes.append(node) 
            self.nodes_low=num.array(nodes) 
            self.nodes_up =num.array(nodes)+gap  # it would be convenient for plot the binning picture 
        elif node_method=='nearby':
        #    print('produce the nodes with with method of nearby')
            if sep is None: 
                seps=num.array([self.mjd[i+1]-self.mjd[i] for i in range(len(self.mjd)-1)])
                seps.sort() 
                sepmin=seps[0] 
                for i in range(len(seps)):
                    if seps[i]>sepmin*Nsep: 
                        if i==len(seps)-1:
                            raise Exception('ALl the seps are within the range of Nsep*sepmin') 
                        break 
                    else:
                        sepmin=seps[i]
                sep=num.average(seps[0:i])  #NOTE would 'maximum' be better here ?

            nodes_low, nodes_up=[], []
            seps=num.array([self.mjd[i+1]-self.mjd[i] for i in range(len(self.mjd)-1)]) #NOTE the above array of seps have been sorted, here we reproduce ...
            nodes_low.append(self.mjd[0])
            for i in range(len(seps)): 
                if seps[i]>sep*Nsep:
                    nodes_up.append(self.mjd[i]) 
                    nodes_low.append(self.mjd[i+1]) 
            nodes_up.append(self.mjd[-1]) 

            self.nodes_low=num.array(nodes_low) 
            self.nodes_up =num.array(nodes_up) 
        else: 
            raise Exception('No such node-method: %s ; only gap and nearby were supported '%node_method)
        return self.nodes_low, self.nodes_up 
    
    def plot_res(self, invert_yaxis=True):
        ax=plt.axes() 
        ax.errorbar(self.mjd ,self.flux ,yerr=self.ferr ,fmt='bo',alpha=0.1)  
        ax.errorbar(self.bmjd,self.bflux,yerr=self.bferr,fmt='bs',capsize=3)
        ylim=plt.ylim()
        x=num.arange(self.mjd[0],self.mjd[-1],0.01) #NOTE When the gap between mjd less than 0.01, the data point in the plot may not the shadow region. If we set 0.001, then it will cost much time to plot the figure, please improve this sentence
        for node_low,node_up in zip(self.nodes_low,self.nodes_up) :
            ax.fill_between(x,ylim[0],ylim[1],where=( (x>=node_low)&(x<=node_up)),color='g',alpha=0.3)
        plt.show()
    
    def binning(self, method='median', Nmin=2, node_method='gap', gap=1,Nsep=5,sep=None, prod_nodes=True): 
        #The method is what to used for binned the data: median or average  
        if prod_nodes==True:
            self.prod_nodes(node_method=node_method,gap=gap, Nsep=Nsep,sep=sep) 

        bmjds=[]; bfluxs=[]; bferrs=[] ; bNs=[]
        for node_low,node_up in zip(self.nodes_low,self.nodes_up):
            ind =(self.mjd>=node_low) & (self.mjd<=node_up)  
        
            if method=='median':
                flux=num.median(self.flux[ind]) 
                if len(self.mjd[ind])>=Nmin:  #Here maybe we can demand more points N, if length< N, we can use the median error of the points 
                    ferr=1.2533*num.std(self.flux[ind],ddof=1) 
                else: 
                    ferr=num.median(self.ferr[ind]) 
            elif method=='average':
                flux=num.mean(self.flux[ind]) 
                if len(self.mjd[ind])>=Nmin: 
                    ferr=num.std(self.flux[ind],ddof=1)
                else: 
                    ferr=num.median(self.ferr[ind]) #Here we can used the propagation of error 
            elif method=='weighted':
                flux=sum( self.flux[ind]/self.ferr[ind] )/ sum( 1/self.ferr[ind]) 
                ferr=len( self.flux[ind])**0.5 / sum( 1/self.ferr[ind] )
            elif method=='errmedian': 
                flux=num.median(self.flux[ind])
                ferr=num.median(self.ferr[ind])
            else: 
                raise Exception('No such method')

            bmjds.append(num.mean(self.mjd[ind])); bfluxs.append(flux); bferrs.append(ferr) ; bNs.append( len(self.mjd[ind]))
        self.bmjd=num.array(bmjds);self.bflux=num.array(bfluxs);self.bferr=num.array(bferrs); self.bN=num.array(bNs)

        if self.show: 
            self.plot_res() 
        return self.bmjd, self.bflux, self.bferr 
    
