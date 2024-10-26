import modal
import os
import numpy as np
import pandas as pd
import glob
from datetime import datetime  # Add this import

app = modal.App("rso-app")

# Use a single volume for parameters
params_volume = modal.Volume.from_name("params-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("wget", "git")  # Add CUDA toolkit
    .pip_install(
        "numpy",
        "pandas",
        "biopython",
        "jax[cuda]",  # Add this line to install CUDA-enabled JAX
        "git+https://github.com/sokrypton/ColabDesign.git",
    )
    .run_commands([
        "mkdir -p /root/params",
        "wget -P /root/params/ https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        "tar -xvf /root/params/alphafold_params_2022-12-06.tar -C /root/params/",
        "rm /root/params/alphafold_params_2022-12-06.tar"
    ])
)

# GPUS: "H100", "A100-80GB", "A10G", "L4", "T4"

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    mounts=[
        modal.Mount.from_local_dir('.', remote_path='/root')
    ]
)
def ros(pdb,trajiters,binderlen):
        # Import colabdesign modules here
    from colabdesign import mk_afdesign_model, clear_mem
    from colabdesign.mpnn import mk_mpnn_model
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants

    print("Imported modules")

    def add_rg_loss(self, weight=0.1):
        '''add radius of gyration loss'''
        def loss_fn(inputs, outputs):
            xyz = outputs["structure_module"]
            ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]

            ca = ca[-self._binder_len:]

            rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
            rg_th = 2.38 * ca.shape[0] ** 0.365
            rg = jax.nn.elu(rg - rg_th)
            return {"rg":rg}
        self._callbacks["model"]["loss"].append(loss_fn)
        self.opt["weights"]["rg"] = weight

    # Remove all PDB files with 'binder_design' in the file name
    [os.remove(pdb_file) for pdb_file in glob.glob('*binder_design*.pdb')]

    clear_mem()
    af_model = mk_afdesign_model(protocol="binder")
    add_rg_loss(af_model)
    af_model.prep_inputs(pdb_filename=pdb, chain='A', hotspot=None, binder_len=binderlen)

    af_model.restart(mode=["gumbel","soft"])
    af_model.set_weights(helix=-0.2, plddt=0.1, pae=0.1,rg=0.5,i_pae=5.0,i_con=2.0)
    af_model.design_logits(trajiters) 
    af_model.save_pdb("backbone.pdb")

    ### SEQ DESIGN AND FILTER ####
    binder_model = mk_afdesign_model(protocol="binder",use_multimer=True,use_initial_guess=True)
    monomer_model = mk_afdesign_model(protocol="fixbb")
    #binder_model.set_weights(i_pae=1.0)
    #print("weights",binder_model.opt["weights"]) 
    #binder_model.opt["weights"]["con"] = 0.0


    mpnn_model = mk_mpnn_model(weights="soluble")
    mpnn_model.prep_inputs(pdb_filename="backbone.pdb", chain='A,B', fix_pos='A',rm_aa="C")

    samples = mpnn_model.sample_parallel(8,temperature=0.01)
    monomer_model.prep_inputs(pdb_filename="backbone.pdb", chain='B')
    binder_model.prep_inputs(pdb_filename="backbone.pdb", chain='A', binder_chain='B',use_binder_template=True,rm_template_ic=True)
    results_df = pd.DataFrame()
    for j,seq in enumerate(samples['seq']):
        #print("Predicting binder only")
        monomer_model.predict(seq=seq[-binderlen:], num_recycles=3)
        if monomer_model.aux['losses']['rmsd'] < 2.0 :
            binder_model.predict(seq=seq[-binderlen:], num_recycles=3)
            #if plddt1 < 0.15 and i_pae < 0.4:
            if True:
                binder_model.save_pdb(f'binder_design_{j}.pdb')
                results_df.loc[j,'seq'] = seq[-binderlen:]
                for key in binder_model.aux["log"]:
                    results_df.loc[j,key] = binder_model.aux["log"][key]
                for weight in af_model.opt["weights"]:
                    results_df.loc[j,f'weights_{key}'] = weight

    # Return the data of the saved PDB file
    return results_df,[open(pdb_file, "rb").read() for pdb_file in glob.glob('*binder_design*.pdb')]

@app.local_entrypoint()
def main(pdb: str, numdesigns: int,trajiters: int,binderlen:int):
    # Upload the PDB file to the shared volume
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{pdb.split('.')[0]}_{date}_binder_designs/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    scores_df = pd.DataFrame()
    for bb_num in range(numdesigns):
        print(f"Starting design for {pdb}")
        
        results_df,out_pdb_data_list = ros.remote(pdb=pdb,trajiters=trajiters,binderlen=binderlen)
        out_pdb_paths = []
        out_pdb_ids = []
        if len(out_pdb_data_list) > 0:
            for i,out_pdb_data in enumerate(out_pdb_data_list):
                    out_pdb = f"{pdb.split('.')[0]}_{date}_binder_design_{bb_num}_{i}.pdb"
                    with open(out_dir+out_pdb, 'wb') as out:
                        out.write(out_pdb_data)
                    out_pdb_paths.append(out_dir+out_pdb)
                    out_pdb_ids.append(out_pdb.split('.')[0])
            results_df['pdb_path'] = out_pdb_paths
            results_df.insert(1, 'pdb_id', out_pdb_ids)
            scores_df = pd.concat([scores_df,results_df]).reset_index(drop=True)
            scores_df.to_csv(out_dir+f"{pdb.split('.')[0]}_{date}_binder_design_scores.csv",index=False)
            print(f"Design completed. PDB files saved at: {out_dir}")

if __name__ == "__main__":
    app.run()
