"""Recount retained outcomes and reconstruct summary statistics; no inference/training."""
from pathlib import Path, PureWindowsPath
import collections,csv,hashlib,json,math,statistics,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obj(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-14),(a,b)

def audit_continued():
    folder=R/'dccn/experiments/continued_qat_20261011'
    files=sorted((folder/'final_test').glob('*.npz'))
    assert len(files)==72,('Missing full archive data',len(files),72)
    sums=collections.defaultdict(list); slots=bits=0
    for p in files:
        j=obj(p.with_suffix('.json'));assert j['slot_data_sha256']==sha(p)
        assert j['pdp_mode']=='3gpp_db_to_linear_power'
        with np.load(p,allow_pickle=False) as z:
            labels=z['labels'];assert labels.shape==(j['slots'],j['bits_per_slot'])
            slots+=labels.shape[0];bits+=labels.size
            errors={}
            for arm,v in j['arms'].items():
                pred=z[arm+'__pred'];e=(pred!=labels).sum(axis=1)
                assert np.array_equal(e,z[arm+'__slot_errors'])
                assert int(e.sum())==v['error_bits'];close(e.sum()/labels.size,v['ber'])
                errors[arm]=pred!=labels
                sums[j['tag'],j['training_seed'],arm].append(float(e.sum()/labels.size))
            for key,c in j['contrasts'].items():
                cand,ref=key.split('_minus_')
                corrected=(errors[ref]&~errors[cand]).sum(axis=1)
                harmed=(errors[cand]&~errors[ref]).sum(axis=1)
                flips=(z[cand+'__pred']!=z[ref+'__pred']).sum(axis=1)
                for suffix,values in [('corrected',corrected),('harmed',harmed),('flips',flips)]:assert np.array_equal(values,z[key+'__'+suffix])
                assert int(corrected.sum())==c['corrected_decisions'] and int(harmed.sum())==c['harmed_decisions']
                close((harmed.sum()-corrected.sum())/labels.size,c['candidate_minus_reference_ber'])
    summary=obj(folder/'summary/summary.json');assert slots==summary['unique_slots']==144000 and bits==summary['unique_bits']==76800000
    grids={k:statistics.mean(v) for k,v in sums.items()}
    assert len(grids)==45 and all(len(v)==8 for v in sums.values())
    primary=[]
    with (folder/'summary/config_contrast_summary.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            ds=[grids[row['tag'],s,row['candidate']]-grids[row['tag'],s,row['reference']] for s in [1,2,3]]
            close(statistics.mean(ds),row['mean_delta_ber']);close(statistics.stdev(ds),row['sample_sd_delta_ber'])
            for s,d in enumerate(ds,1):close(d,row[f'seed{s}_delta_ber'])
            if row['contrast']=='A1_minus_A2':primary.append({'tag':row['tag'],'mean_delta_ber':statistics.mean(ds),'sample_sd':statistics.stdev(ds),'individual_seed_deltas':ds})
    manifest=obj(folder/'summary/manifest.json')
    for n,row in manifest['outputs'].items():assert sha(folder/'summary'/n)==row['sha256']
    return {'conditions':len(files),'unique_slots':slots,'unique_bits':bits,'primary':primary,'all_config_contrasts_checked':18}

def audit_interventions(rel,precision=False):
    folder=R/rel;files=sorted(folder.glob('*.npz'));assert len(files)==72,(rel,len(files))
    variants=0
    for p in files:
        j=obj(p.with_suffix('.json'));assert sha(p)==j['slot_data_sha256']
        with np.load(p,allow_pickle=False) as z:
            baseline=z['baseline_slot_errors']
            for name,v in j['variants'].items():
                errors=z[name+'__slot_errors'];assert errors.shape==baseline.shape==(j['slots'],)
                expected=v['errors'] if precision else v['candidate_error_bits'];assert int(errors.sum())==expected
                if precision:
                    fixed=z[name+'__baseline_only_errors'];new=z[name+'__variant_only_errors']
                else:fixed=z[name+'__corrected'];new=z[name+'__harmed']
                assert np.array_equal(errors-baseline,new-fixed)
                variants+=1
    return {'conditions':len(files),'variant_condition_counts_checked':variants}

def audit_transfer():
    folder=R/'e5_hls/experiments/local_a12_20261002/transfer_audit'
    files=sorted(folder.glob('*dB.npz'));assert len(files)==8
    totals=collections.Counter()
    for p in files:
        j=obj(p.with_suffix('.json'));assert sha(p)==j['npz_sha256']
        with np.load(p,allow_pickle=False) as z:
            for name,count in j['errors'].items():
                errors=(z[name+'_pred']!=z['labels']).sum(axis=1)
                assert np.array_equal(errors,z[name+'_errors']) and int(errors.sum())==count
                totals[name]+=count
    net_c=totals['c_rev_t']-totals['c_a12'];net_t=totals['torch_rev_t']-totals['torch_a12']
    assert net_c==1916 and net_t==1919
    closure=obj(folder.parent/'closure.json');a=closure['physical']['local_a12']
    assert a['bram18_equivalents']==2*a['ramb36']+a['ramb18']==274
    assert closure['hls']['local_a12']['latency']==58115
    return {'conditions':8,'error_totals':dict(totals),'c_net_gain':net_c,'torch_net_gain':net_t,'retained_fraction':net_c/net_t,'hls_latency_cycles':58115,'bram18_equivalents':274}

def audit_components():
    folder=R/'dccn/experiments/component_ablation_20261001/common_budget_7arm'
    summary=obj(folder/'summary.json')['summary']['channels'];values=collections.defaultdict(list)
    points=pairs=0
    for seed in [1,2,3]:
        row=obj(folder/f'seed{seed}.json');assert row['status']=='COMMON_BUDGET_RETRAINED_COMPONENT_ABLATION'
        for channel,curve in row['evaluation'].items():
            for p in curve['points']:
                points+=1;assert len(p['errors'])==7
                for arm,n in p['errors'].items():
                    close(n/p['bits'],p['ber'][arm]);values[channel,arm,str(float(p['snr_db']))].append(p['ber'][arm])
                for arm,v in p['paired_vs_slim2'].items():
                    c=v['bit_contingency'];assert sum(c.values())==p['bits']
                    diff=p['errors'][arm]-p['errors']['slim2']
                    assert c['variant_only_error']-c['base_only_error']==diff==sum(v['per_slot_error_count_delta'])
                    close(diff/p['bits'],v['variant_minus_base_ber']);pairs+=1
    assert points==48 and pairs==288
    for (channel,arm,snr),observations in values.items():
        v=summary[channel][arm]['at_snr'][snr]['ber']
        close(statistics.mean(observations),v['mean']);close(statistics.stdev(observations),v['sample_sd'])
    return {'conditions':points,'arms':7,'paired_count_conservation_checks':pairs,'grid_mean_sample_sd_checks':len(values)}

def audit_baseline_counts():
    points=0;bound_checkpoints=0
    for seed in [1,2,3]:
        row=obj(R/f'dccn/results/paired_b1_longcp_pdpfix_seed{seed}.json')['seed_result']
        for name,digest in row['checkpoint_sha256'].items():
            file=PureWindowsPath(row['checkpoint_paths'][name]).name
            assert sha(R/'dccn/ckpt'/file)==digest;bound_checkpoints+=1
        for curve in row['channels'].values():
            for p in curve['points']:
                for name,n in p['errors'].items():close(n/p['bits'],p['ber'][name])
                points+=1
    assert points==48 and bound_checkpoints==12
    return {'conditions':points,'checkpoint_bindings':bound_checkpoints,'scope':'Stored aggregate error/BER and checkpoint-hash consistency; no raw traffic regenerated.'}

def main():
    if not (R/'dccn/experiments/continued_qat_20261011/final_test').is_dir():
        print(json.dumps({'status':'missing_full_archive','message':'Extract the complete Zenodo archive before running raw-data audits. GitHub contains code and light summaries only.'}));sys.exit(2)
    report={'status':'pass','scope':'Recorded-data recount only; no new traffic, retraining, bootstrap resampling or hardware build.',
      'continued_qat':audit_continued(),
      'precision_screen':audit_interventions('dccn/experiments/precision_localization_20261001/screen_2000',True),
      'back_decomposition':audit_interventions('dccn/experiments/back_precision_followup_20261002/independent_2000'),
      'a12_confirmation':audit_interventions('dccn/experiments/back_precision_followup_20261002/confirmation_2000'),
      'native_c_transfer':audit_transfer(),'components':audit_components(),'baseline_counts':audit_baseline_counts()}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
