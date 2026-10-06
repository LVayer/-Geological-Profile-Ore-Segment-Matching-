"""Real widget + selected-method job integration tests, not just UI construction."""
import threading
import time
import tkinter as tk
import pytest
from stratamatch.io import read,local,write
from stratamatch.jobs import run_job,validate_request,Cancelled,import_image
from stratamatch.engine import compare
from stratamatch.synthetic import make_case
from stratamatch.gui import Application,SectionEditor,PairReviewWindow,numbers

def request():return read('data/synthetic/images/split/manifest.json')['sections'],read('config/default.json')

@pytest.fixture(scope='module')
def tk_root():
    root=tk.Tk();root.withdraw()
    yield root
    root.destroy()

def test_selected_methods_and_export_without_graph(tmp_path):
    sections,config=request();result=run_job(sections,['dtw'],config,tmp_path)
    assert result['status']=='complete'
    report=read(local(result['directory'])/'pair_000/matches.json')
    assert list(report['methods'])==['dtw']
    assert (local(result['directory'])/'pair_000/matching_dtw.png').is_file()
    manifest=read(local(result['directory'])/'pair_000/review_manifest.json')
    assert manifest['schema']=='stratamatch-review-v1'
    assert (local(result['directory'])/'pair_000'/manifest['overlays']['dtw']['source']).is_file()
    assert (local(result['directory'])/'pair_000'/manifest['overlays']['dtw']['target']).is_file()
    assert local(result['reviews'][0]).is_file()
    assert local(result['pages'][0]).is_file()
    assert read(local(result['directory'])/'request.json')['methods']==['dtw']

def test_jobs_do_not_overwrite_and_honor_cancellation(tmp_path):
    sections,config=request();event=threading.Event();event.set()
    with pytest.raises(Cancelled):run_job(sections,['graph'],config,tmp_path,cancel=event)
    dirs=list(tmp_path.iterdir());assert read(dirs[0]/'status.json')['status']=='cancelled'
    result=run_job(sections,['graph'],config,tmp_path);assert local(result['directory'])!=dirs[0]

def test_gui_request_checks():
    sections,config=request()
    with pytest.raises(ValueError):validate_request(sections,[],config,'outputs/gui')
    with pytest.raises(ValueError):validate_request(sections,['graph'],config,'../outside')
    sections[1]['station']=0
    with pytest.raises(ValueError):validate_request(sections,['graph'],config,'outputs/gui')

def test_request_rejects_images_with_only_unconfirmed_ocr_labels():
    sections,config=request()
    for entry in sections[0]['legend']:
        entry['verified']=False;entry['lithology']='UNKNOWN'
    with pytest.raises(ValueError,match='没有人工确认的岩性图例'):
        validate_request(sections,['graph'],config,'outputs/gui')

def test_empty_method_selection_is_not_all():
    c,_=make_case('continuous')
    with pytest.raises(ValueError):compare(c['source'],c['target'],methods=[])

def test_gui_widgets_run_and_editor_preserves_metadata(tmp_path,monkeypatch,tk_root):
    root=tk_root;app=Application(root);errors=[];monkeypatch.setattr(app,'error',lambda e:errors.append(str(e)))
    try:
        app.load_demo();assert len(app.tree.get_children())==2
        saved=[];editor=SectionEditor(root,app.sections[0],saved.append);root.update()
        editor.vars['station'].set('12.5');editor.save();assert saved[0]['station']==12.5 and saved[0]['legend']
        for m,v in app.methods.items():v.set(m=='markov')
        app.output.set(str(tmp_path));app.start()
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and (app.worker.is_alive() or not app.last_result):
            root.update();time.sleep(.02)
            if errors:break
        assert not errors and app.last_result and len(app.result_tree.get_children())==1
        report=read(local(app.last_result['directory'])/'pair_000/matches.json')
        assert list(report['methods'])==['markov']
        review=PairReviewWindow(root,app.last_result['reviews'][0]);root.update();assert review.rows();review.destroy()
    finally:app.destroy()

def test_saved_gui_configuration_reload(tmp_path,tk_root):
    sections,cfg=request();cfg['lab_tolerance']=8
    path=tmp_path/'project.json';write(path,{'sections':sections,'ui_config':cfg,'methods':['graph'],'override_recognition':True,'output':str(tmp_path)})
    root=tk_root;app=Application(root)
    try:
        app.load(path);assert app.params['lab_tolerance'].get()=='8'
        assert app.override.get() and app.methods['graph'].get() and not app.methods['dtw'].get()
        assert app.output.get()==str(tmp_path)
    finally:app.destroy()

def test_auto_import_background_and_review_controls(tk_root,monkeypatch):
    import stratamatch.ocr as ocr
    monkeypatch.setattr(ocr,'recognize_text',lambda *args:{'status':'available','lines':[{'words':[{'text':'shale','box':[186,10,25,8]}]}]})
    app=Application(tk_root);errors=[];monkeypatch.setattr(app,'error',lambda e:errors.append(str(e)))
    try:
        app.import_files([str(local('data/synthetic/images/continuous/A.png'))])
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and not app.sections:tk_root.update();time.sleep(.02)
        assert not errors and app.sections and len(app.sections[0]['legend'])==3
        s=app.sections[0];assert s['roi']==[4,10,152,88] and not s['layout_reviewed']
        editor=SectionEditor(tk_root,s,lambda _:None)
        assert 'layout_reviewed' in editor.vars and not editor.vars['layout_reviewed'].get()
        editor.legend_tree.selection_set('0')
        monkeypatch.setattr('stratamatch.gui.simpledialog.askstring',lambda *a,**k:'页岩')
        editor.confirm_legend();assert editor.legend[0]['lithology']=='页岩' and editor.legend[0]['verified']
        editor.destroy()
    finally:app.destroy()
