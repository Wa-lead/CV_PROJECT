import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem




class ChemCeptionizer():
    
    """
    This is the featurizer for the ChemCeption model. it takes a molecule and returns a 2D mol of the molecule of 4 channels:
    1. Bond order
    2. Atom type
    3. Atom charge
    4. Atom hybridization
    """
    def __init__(self, embed: float=12.0, res: float=0.5, fuse=True):
        self.embed = embed
        self.res = res
        self.fuse = fuse

    def featurize(self, mol):
        try:
            dims = int(self.embed*2/self.res)
            cmol = Chem.Mol(mol.ToBinary())
            cmol.ComputeGasteigerCharges()
            AllChem.Compute2DCoords(cmol)
            coords = cmol.GetConformer(0).GetPositions()
            vect = np.zeros((dims,dims,4))
            #Bonds first
            for i,bond in enumerate(mol.GetBonds()):
                bondorder = bond.GetBondTypeAsDouble()
                bidx = bond.GetBeginAtomIdx()
                eidx = bond.GetEndAtomIdx()
                bcoords = coords[bidx]
                ecoords = coords[eidx]
                frac = np.linspace(0,1,int(1/self.res*2)) #
                for f in frac:
                    c = (f*bcoords + (1-f)*ecoords)
                    idx = int(round((c[0] + self.embed)/self.res))
                    idy = int(round((c[1]+ self.embed)/self.res))
                    #Save in the vector first channel
                    vect[ idx , idy ,0] = bondorder
            #Atom Layers
            for i,atom in enumerate(cmol.GetAtoms()):
                    idx = int(round((coords[i][0] + self.embed)/self.res))
                    idy = int(round((coords[i][1]+ self.embed)/self.res))
                    #Atomic number
                    vect[ idx , idy, 1] = atom.GetAtomicNum()
                    #Gasteiger Charges
                    charge = atom.GetProp("_GasteigerCharge")
                    vect[ idx , idy, 3] = charge
                    #Hybridization
                    hyptype = atom.GetHybridization().real
                    vect[ idx , idy, 2] = hyptype
            
            if self.fuse:
                vect = self.weighted_fusion(vect)
                return vect
            else:
                return vect
        except:
            return None
        
    def weighted_fusion(self, mol, channel_4_weight=0.1):
        
        """
        This is to change mol from 4 channels to 3 channels
        the reason is to use other pretrained model
        """
        mol = mol.astype(np.float32)/255
        
        base_channel_weight = 1 - channel_4_weight
        
        new_channel_1 =  base_channel_weight* mol[:, :, 0] + channel_4_weight * mol[:, :, 3]
        new_channel_2 =  base_channel_weight* mol[:, :, 1] + channel_4_weight * mol[:, :, 3]
        new_channel_3 =  base_channel_weight* mol[:, :, 2] + channel_4_weight * mol[:, :, 3]

        # normalize
        new_channel_1 = (new_channel_1 - np.min(new_channel_1)) / (np.max(new_channel_1) - np.min(new_channel_1))
        new_channel_2 = (new_channel_2 - np.min(new_channel_2)) / (np.max(new_channel_2) - np.min(new_channel_2))
        new_channel_3 = (new_channel_3 - np.min(new_channel_3)) / (np.max(new_channel_3) - np.min(new_channel_3))
        
        # Create a new 3-channel mol
        merged_mol = np.stack((new_channel_1, new_channel_2, new_channel_3), axis=-1)
        merged_mol = merged_mol.astype(np.float32)
        
        return merged_mol        