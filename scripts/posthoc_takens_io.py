import zipfile, numpy as np
from numpy.lib import format as npf
def membro(caminho, nome):
    z=zipfile.ZipFile(caminho); info=z.getinfo(nome)
    f=open(caminho,'rb'); f.seek(info.header_offset); h=f.read(30)
    n,e=int.from_bytes(h[26:28],'little'),int.from_bytes(h[28:30],'little')
    f.seek(info.header_offset+30+n+e)
    v=npf.read_magic(f); shape,fo,dt=(npf.read_array_header_1_0 if v==(1,0) else npf.read_array_header_2_0)(f)
    return np.memmap(caminho,dtype=dt,mode='r',offset=f.tell(),shape=shape,order='F' if fo else 'C')
def abrir(c):
    return {k:membro(c,k+'.npy') for k in ('ecg_id','sinais','arestas','offsets')}
