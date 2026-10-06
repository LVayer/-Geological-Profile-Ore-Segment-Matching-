"""Native local workstation. Tk widgets stay on the main thread; jobs use a queue."""
from copy import deepcopy
from pathlib import Path
import os
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox,simpledialog
from PIL import Image,ImageTk
from .io import ROOT,local,read,write
from .engine import METHODS
from .jobs import import_image,run_job,validate_request,Cancelled
from .layout import detect,apply_proposal,attach_text,prepare_image,legend_region,annotation_boxes

def configure_dpi():
    """Use real window coordinates on Windows with 125%/150% display scaling."""
    if os.name=='nt':
        import ctypes
        try:ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError,AttributeError):pass

def numbers(text,kind=float,count=4):
    values=[kind(v.strip()) for v in text.replace('，',',').split(',')]
    if len(values)!=count:raise ValueError(f'请输入 {count} 个以逗号分隔的数字')
    return values

class SectionEditor(tk.Toplevel):
    """Edit metadata and pick plot/legend rectangles in original image coordinates."""
    def __init__(self,parent,section,on_save):
        super().__init__(parent);self.title('剖面设置 · 绘图区与图例');self.geometry('1100x780');self.minsize(900,680)
        self.data=deepcopy(section);self.on_save=on_save;self.vars={};self.drag=None;self.scale=1.;self.offset=(0,0);self.outline_points=[]
        self.transient(parent);self.grab_set()
        top=ttk.Frame(self,padding=12);top.pack(fill='x')
        fields=[('id','剖面编号'),('station','空间位置 station'),('frame','共同坐标基准'),('units','物理单位'),('bounds','物理范围 xmin,ymin,xmax,ymax')]
        if 'layers' not in section and 'instance_mask' not in section:fields+=[('roi','绘图区 x,y,宽,高'),('expected_layer_count','人工粗略地层数（仅校验，可留空）')]
        for i,(key,label) in enumerate(fields):
            row,col=divmod(i,3);ttk.Label(top,text=label).grid(row=row*2,column=col,sticky='w',padx=6)
            val=section.get(key,'');val=','.join(map(str,val)) if isinstance(val,list) else str(val)
            var=tk.StringVar(value=val);self.vars[key]=var
            ttk.Entry(top,textvariable=var,width=34).grid(row=row*2+1,column=col,sticky='ew',padx=6,pady=(0,8));top.columnconfigure(col,weight=1)
        checks=ttk.Frame(self,padding=(18,0));checks.pack(fill='x')
        flags=[('registration_verified','已确认坐标配准'),('complete','已确认地层覆盖完整'),('complex_structure','复杂断层/褶皱')]
        if 'image' in section and 'layers' not in section:flags.append(('layout_reviewed','已核对主图/图例位置'))
        for key,label in flags:
            default=key=='layout_reviewed' and 'auto_layout' not in section
            var=tk.BooleanVar(value=bool(section.get(key,default)));self.vars[key]=var;ttk.Checkbutton(checks,text=label,variable=var).pack(side='left',padx=5)
        self.legend=deepcopy(section.get('legend',[]));self.mode=tk.StringVar(value='roi')
        image_path=section.get('image');self.im=None
        if image_path and 'layers' not in section:
            with Image.open(local(image_path)) as source:
                source.seek(section.get('page',0));self.im=source.convert('RGB')
            toolbar=ttk.Frame(self,padding=(14,8));toolbar.pack(fill='x')
            for mode,label in [('roi','框选绘图区'),('legend','框选图例色块'),('outline','逐点勾画内容外轮廓'),('outline_hole','逐点勾画非地层孔洞'),('exclude','框选文字/注释排除区'),('fault','框选断层区域')]:
                ttk.Radiobutton(toolbar,text=label,value=mode,variable=self.mode).pack(side='left',padx=5)
            auto=ttk.Frame(self,padding=(16,2));auto.pack(fill='x')
            ttk.Button(auto,text='自动定位主图/图例',command=self.auto_detect).pack(side='left')
            ttk.Button(auto,text='完成当前轮廓',command=self.finish_outline).pack(side='left',padx=(8,0))
            ttk.Button(auto,text='撤销轮廓点',command=self.undo_outline).pack(side='left',padx=(4,0))
            self.ocr_button=ttk.Button(auto,text='识别图例文字（本地 OCR）',command=self.start_ocr);self.ocr_button.pack(side='left',padx=8)
            self.auto_status=tk.StringVar(value='绿色=主图，青色=内容外缘，黄色=图例，橙色=自动文字杂项。双击图例行核对岩性。')
            ttk.Label(auto,textvariable=self.auto_status,foreground='#405976',wraplength=600).pack(side='left')
            self.canvas=tk.Canvas(self,background='#182637',highlightthickness=0,height=300);self.canvas.pack(fill='both',expand=True,padx=16,pady=8)
            self.canvas.bind('<Configure>',self.draw);self.canvas.bind('<ButtonPress-1>',self.start_drag);self.canvas.bind('<B1-Motion>',self.dragging);self.canvas.bind('<ButtonRelease-1>',self.end_drag)
            self.canvas.bind('<Button-3>',lambda e:self.undo_outline())
        else:
            ttk.Label(self,text='已加载结构化地层或实例掩膜；此处编辑坐标信息，保留原有地层数据。',padding=20).pack(fill='x')
        self.legend_tree=ttk.Treeview(self,columns=('lith','color','verified'),show='headings',height=4)
        for c,t in [('lith','岩性'),('color','色块位置 / RGB'),('verified','已确认')]:self.legend_tree.heading(c,text=t)
        self.legend_tree.pack(fill='x',padx=16);self.refresh_legend();self.legend_tree.bind('<Double-1>',lambda e:self.confirm_legend())
        bottom=ttk.Frame(self,padding=12);bottom.pack(fill='x')
        ttk.Button(bottom,text='删除选中图例',command=self.delete_legend).pack(side='left')
        ttk.Button(bottom,text='修改/确认图例',command=self.confirm_legend).pack(side='left',padx=6)
        ttk.Button(bottom,text='清空排除区 / 断层框',command=self.clear_polygons).pack(side='left',padx=8)
        ttk.Button(bottom,text='清空内容轮廓',command=self.clear_content_outline).pack(side='left',padx=8)
        ttk.Button(bottom,text='保存剖面设置',command=self.save).pack(side='right')
        ttk.Button(bottom,text='取消',command=self.destroy).pack(side='right',padx=8)
        if self.im is not None:
            bottom.pack_configure(side='bottom',before=self.canvas)
            self.legend_tree.pack_configure(side='bottom',before=self.canvas)

    def refresh_legend(self):
        self.legend_tree.delete(*self.legend_tree.get_children())
        for i,e in enumerate(self.legend):self.legend_tree.insert('', 'end',iid=str(i),values=(e.get('lithology'),e.get('swatch',e.get('rgb')), '是' if e.get('verified') else '否'))

    def confirm_legend(self):
        selected=self.legend_tree.selection()
        if not selected:return
        entry=self.legend[int(selected[0])]
        text=simpledialog.askstring('核对岩性','核对图例文字，输入完整且统一的岩性名称：',initialvalue=entry.get('lithology',''),parent=self)
        if text and text.strip() and text.strip()!='UNKNOWN':entry['lithology']=text.strip();entry['verified']=True;self.refresh_legend()

    def auto_detect(self):
        try:
            current=deepcopy(self.data);current['legend']=self.legend
            if 'layout_reviewed' in self.vars:current['layout_reviewed']=self.vars['layout_reviewed'].get()
            proposal=detect(self.data['image'],self.data.get('page',0));updated=apply_proposal(current,proposal)
            if updated.get('roi')!=self.data.get('roi'):self.vars['registration_verified'].set(False)
            self.data=updated;self.legend=updated['legend'];self.vars['roi'].set(','.join(map(str,updated['roi'])))
            blank=sum(e.get('inferred_blank',False) for e in proposal['legend'])
            self.auto_status.set(f'主图候选 {len(proposal["roi_candidates"])} 个，图例候选 {len(proposal["legend"])} 个（含白色/空心 {blank} 个）；青色曲线为实际内容边界')
            self.refresh_legend();self.draw()
        except Exception as exc:messagebox.showerror('自动定位失败',str(exc),parent=self)

    def start_ocr(self):
        if not self.legend:self.auto_status.set('请先自动定位图例或手工框选色块');return
        if getattr(self,'ocr_running',False):return
        from .ocr import recognize_text
        self.ocr_running=True;self.ocr_button.configure(state='disabled');self.auto_status.set('正在本地识别文字…')
        original=deepcopy(self.legend);pending=queue.Queue();image=self.data['image'];page=self.data.get('page',0);region=legend_region(original,self.im.size)
        def work():
            try:pending.put({'legend':recognize_text(image,page,region),'plot':recognize_text(image,page,numbers(self.vars['roi'].get(),int))})
            except Exception as exc:pending.put({'status':'unavailable','reason':str(exc),'lines':[]})
        threading.Thread(target=work,daemon=True).start()
        def poll():
            try:result=pending.get_nowait()
            except queue.Empty:self.ocr_after=self.after(100,poll);return
            self.ocr_after=None;self.ocr_running=False;self.ocr_button.configure(state='normal')
            if original!=self.legend:self.auto_status.set('图例已修改，请重新识别，避免覆盖新设置');return
            if 'legend' in result and result['legend']['status']=='available':
                self.legend=attach_text(self.legend,result['legend']['lines'])
                if result['plot']['status']=='available':self.data['auto_text_boxes']=annotation_boxes(result['plot']['lines'],numbers(self.vars['roi'].get(),int),self.legend)
                self.refresh_legend();self.draw();self.auto_status.set(f'OCR 完成，主图内发现 {len(self.data.get("auto_text_boxes",[]))} 个文字杂项框；请核对橙色框')
            else:self.auto_status.set('本地 OCR 不可用，请双击图例行手工填写岩性')
        self.ocr_after=self.after(100,poll)

    def destroy(self):
        if getattr(self,'ocr_after',None):self.after_cancel(self.ocr_after);self.ocr_after=None
        super().destroy()

    def draw(self,event=None):
        if self.im is None:return
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        self.scale=min(w/self.im.width,h/self.im.height);size=(max(1,int(self.im.width*self.scale)),max(1,int(self.im.height*self.scale)))
        self.offset=((w-size[0])/2,(h-size[1])/2);self.photo=ImageTk.PhotoImage(self.im.resize(size))
        self.canvas.delete('all');self.canvas.create_image(*self.offset,image=self.photo,anchor='nw')
        rectangles=[]
        try:rectangles.append((numbers(self.vars['roi'].get(),int),'#55e7af'))
        except (ValueError,KeyError):pass
        rectangles += [(e['swatch'],'#ffcc66') for e in self.legend if 'swatch' in e]
        for rect,color in rectangles:
            x,y,w,h=rect;ox,oy=self.offset;s=self.scale
            self.canvas.create_rectangle(ox+x*s,oy+y*s,ox+(x+w)*s,oy+(y+h)*s,outline=color,width=2)
        for key,color in [('exclude_polygons','#ff8585'),('fault_polygons','#d4a5ff')]:
            for poly in self.data.get(key,[]):self.canvas.create_polygon(*[c for x,y in poly for c in (self.offset[0]+x*self.scale,self.offset[1]+y*self.scale)],outline=color,fill='',width=2)
        for x,y,w,h in self.data.get('auto_text_boxes',[]):
            ox,oy=self.offset;s=self.scale;self.canvas.create_rectangle(ox+x*s,oy+y*s,ox+(x+w)*s,oy+(y+h)*s,outline='#ff9f43',width=1)
        rings=self.data.get('plot_content_rings') or [{'points':p,'hole':False} for p in self.data.get('plot_content_polygons',[])]
        for ring in rings:
            color='#ff78c6' if ring.get('hole') else '#48d7ff'
            self.canvas.create_polygon(*[c for x,y in ring['points'] for c in (self.offset[0]+x*self.scale,self.offset[1]+y*self.scale)],outline=color,fill='',width=2)
        if self.outline_points:
            colour='#48d7ff' if self.mode.get()=='outline' else '#ff78c6'
            coords=[c for x,y in self.outline_points for c in (self.offset[0]+x*self.scale,self.offset[1]+y*self.scale)]
            if len(coords)>=4:self.canvas.create_line(*coords,fill=colour,width=3)
            for x,y in self.outline_points:self.canvas.create_oval(self.offset[0]+x*self.scale-3,self.offset[1]+y*self.scale-3,self.offset[0]+x*self.scale+3,self.offset[1]+y*self.scale+3,fill=colour,outline='white')

    def _image_point(self,event):
        ox,oy=self.offset;s=self.scale
        return [max(0,min(self.im.width,round((event.x-ox)/s))),max(0,min(self.im.height,round((event.y-oy)/s)))]

    def start_drag(self,event):
        if self.mode.get() in ('outline','outline_hole'):
            point=self._image_point(event)
            if not self.outline_points or sum((a-b)**2 for a,b in zip(point,self.outline_points[-1]))>=4:self.outline_points.append(point)
            self.drag=None;self.draw();return
        self.drag=(event.x,event.y)
    def dragging(self,event):
        if self.mode.get() in ('outline','outline_hole'):return
        if self.drag:self.canvas.delete('selection');self.canvas.create_rectangle(*self.drag,event.x,event.y,outline='white',width=2,tags='selection')
    def end_drag(self,event):
        if self.mode.get() in ('outline','outline_hole'):return
        if not self.drag:return
        x0,y0=self.drag;self.drag=None;ox,oy=self.offset;s=self.scale
        x1=max(0,min(self.im.width,round((min(x0,event.x)-ox)/s)));x2=max(0,min(self.im.width,round((max(x0,event.x)-ox)/s)))
        y1=max(0,min(self.im.height,round((min(y0,event.y)-oy)/s)));y2=max(0,min(self.im.height,round((max(y0,event.y)-oy)/s)))
        if x2-x1<2 or y2-y1<2:return
        mode=self.mode.get();rect=[x1,y1,x2-x1,y2-y1]
        if mode=='roi':
            self.vars['roi'].set(','.join(map(str,rect)));self.vars['registration_verified'].set(False)
            if 'layout_reviewed' in self.vars:self.vars['layout_reviewed'].set(True)
        elif mode=='legend':
            lith=simpledialog.askstring('图例岩性','输入该色块对应的标准岩性名称（确定后视为已确认）：',parent=self)
            if lith and lith.strip():self.legend.append({'lithology':lith.strip(),'swatch':rect,'verified':True});self.refresh_legend()
        else:self.data.setdefault('fault_polygons' if mode=='fault' else 'exclude_polygons',[]).append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        self.draw()
    def finish_outline(self):
        if self.mode.get() not in ('outline','outline_hole'):
            messagebox.showinfo('轮廓勾画','请先选择“逐点勾画内容外轮廓”或“逐点勾画非地层孔洞”。',parent=self);return
        if len(self.outline_points)<3:
            messagebox.showerror('轮廓点不足','闭合轮廓至少需要 3 个点；曲线位置请增加采样点。',parent=self);return
        hole=self.mode.get()=='outline_hole';ring={'points':deepcopy(self.outline_points),'hole':hole}
        rings=deepcopy(self.data.get('plot_content_rings',[]))
        if hole:rings.append(ring)
        else:rings=[existing for existing in rings if existing.get('hole')];rings.insert(0,ring)
        self.data['plot_content_rings']=rings
        self.data['plot_content_polygons']=[r['points'] for r in rings if not r.get('hole')]
        self.outline_points=[]
        if 'layout_reviewed' in self.vars:self.vars['layout_reviewed'].set(True)
        self.auto_status.set('内容轮廓已保存：青色为外轮廓，粉色为明确非地层孔洞；识别不会越过外轮廓。')
        self.draw()
    def undo_outline(self):
        if self.outline_points:self.outline_points.pop();self.draw()
    def clear_content_outline(self):
        self.outline_points=[];self.data['plot_content_rings']=[];self.data['plot_content_polygons']=[];self.draw()
    def delete_legend(self):
        for index in sorted((int(v) for v in self.legend_tree.selection()),reverse=True):self.legend.pop(index)
        self.refresh_legend();self.draw()
    def clear_polygons(self):self.data['exclude_polygons']=[];self.data['fault_polygons']=[];self.draw()
    def save(self):
        try:
            for k,v in self.vars.items():
                val=v.get()
                if k=='expected_layer_count':self.data[k]=int(val) if str(val).strip() else None
                else:self.data[k]=numbers(val,int if k=='roi' else float) if k in ['roi','bounds'] else (float(val) if k=='station' else val)
            if not self.data['id'].strip():raise ValueError('剖面编号不能为空')
            if self.im:self.data['legend']=self.legend
            self.on_save(self.data);self.destroy()
        except Exception as exc:messagebox.showerror('设置无效',str(exc),parent=self)

class PairReviewWindow(tk.Toplevel):
    """Review algorithm rows as two-column buttons and save human decisions."""
    def __init__(self,parent,manifest_path):
        super().__init__(parent);self.title('地层对应人工审查');self.geometry('1400x900');self.minsize(1050,700)
        self.manifest_path=local(manifest_path);self.folder=self.manifest_path.parent;self.manifest=read(self.manifest_path)
        self.report=read(self.folder/self.manifest['matches_file']);sections=read((self.folder/self.manifest['sections_file']).resolve())['sections']
        by_id={s['id']:s for s in sections};self.sa=by_id[self.manifest['source_section']];self.sb=by_id[self.manifest['target_section']]
        self.method=tk.StringVar(value=next(iter(self.report['methods'])));self.selected=None;self.photos=[];self.custom={m:[] for m in self.report['methods']};self.decisions={m:{} for m in self.report['methods']}
        existing=self.folder/'manual_review.json'
        if existing.is_file():
            saved=read(existing);self.decisions.update(saved.get('decisions',{}));self.custom.update(saved.get('custom_rows',{}))
        top=ttk.Frame(self,padding=10);top.pack(fill='x');ttk.Label(top,text='方法').pack(side='left')
        combo=ttk.Combobox(top,textvariable=self.method,values=list(self.report['methods']),state='readonly',width=14);combo.pack(side='left',padx=8);combo.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        ttk.Label(top,text='同色=对应；同色系=同岩性不同层；条纹=merge；灰色=目标侧未对应。',foreground='#405976').pack(side='left',padx=12)
        images=ttk.Frame(self);images.pack(fill='x',padx=10)
        self.source_image=ttk.Label(images,anchor='center');self.source_image.pack(side='left',fill='both',expand=True)
        self.target_image=ttk.Label(images,anchor='center');self.target_image.pack(side='left',fill='both',expand=True,padx=(10,0))
        body=ttk.Frame(self);body.pack(fill='both',expand=True,padx=10,pady=8)
        canvas=tk.Canvas(body,highlightthickness=0);scroll=ttk.Scrollbar(body,orient='vertical',command=canvas.yview);scroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        self.rows_frame=ttk.Frame(canvas);window=canvas.create_window(0,0,window=self.rows_frame,anchor='nw')
        self.rows_frame.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')));canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width));canvas.configure(yscrollcommand=scroll.set)
        bottom=ttk.Frame(self,padding=10);bottom.pack(fill='x')
        for text,status in [('确认对应','confirmed'),('标记不对应','rejected'),('仍需复核','uncertain')]:ttk.Button(bottom,text=text,command=lambda s=status:self.set_decision(s)).pack(side='left',padx=4)
        ttk.Button(bottom,text='新增/修正对应…',command=self.add_custom).pack(side='left',padx=12)
        ttk.Button(bottom,text='保存人工审查',command=self.save_review).pack(side='right',padx=4)
        ttk.Button(bottom,text='导出三维建模输入',command=self.export_modeling).pack(side='right',padx=4)
        self.detail=tk.StringVar(value='点击一行查看证据并进行人工判定。');ttk.Label(self,textvariable=self.detail,padding=(12,4),wraplength=1300).pack(fill='x')
        self.refresh()

    def rows(self):return self.report['methods'][self.method.get()]['results']+self.custom.get(self.method.get(),[])
    def refresh(self):
        method=self.method.get();overlay=self.manifest['overlays'][method];self.photos=[]
        for widget,key in [(self.source_image,'source'),(self.target_image,'target')]:
            with Image.open(self.folder/overlay[key]) as image:
                image=image.convert('RGB');image.thumbnail((650,330),Image.Resampling.LANCZOS);photo=ImageTk.PhotoImage(image.copy())
            self.photos.append(photo);widget.configure(image=photo)
        for child in self.rows_frame.winfo_children():child.destroy()
        ttk.Label(self.rows_frame,text=f'{self.sa["id"]} 基准地层',width=38,anchor='center').grid(row=0,column=0,sticky='ew')
        ttk.Label(self.rows_frame,text=f'{self.sb["id"]} 对比地层',width=38,anchor='center').grid(row=0,column=1,sticky='ew')
        ttk.Label(self.rows_frame,text='关系 / 人工状态',width=28,anchor='center').grid(row=0,column=2,sticky='ew')
        decisions=self.decisions.setdefault(method,{})
        for i,row in enumerate(self.rows(),1):
            key=str(i-1);left=' + '.join(row.get('source_layers',[])) or '（空）';right=' + '.join(row.get('target_layers',[])) or '（空）'
            human=decisions.get(key,'pending');text=f'{left:<34} │ {right:<34} │ {row.get("proposed_relation",row.get("relation_type",""))} / {human}'
            button=tk.Button(self.rows_frame,text=text,anchor='w',font=('Consolas',9),relief='raised',command=lambda n=i-1:self.select_row(n))
            button.grid(row=i,column=0,columnspan=3,sticky='ew',pady=1);self.rows_frame.columnconfigure(0,weight=1)
        self.selected=None

    def select_row(self,index):
        self.selected=index;row=self.rows()[index]
        self.detail.set(json.dumps({k:row.get(k) for k in ['source_layers','target_layers','proposed_relation','confidence','status','reasons','features']},ensure_ascii=False))

    def set_decision(self,status):
        if self.selected is None:return
        self.decisions.setdefault(self.method.get(),{})[str(self.selected)]=status;self.refresh()

    def add_custom(self):
        rows=self.rows();initial_source=','.join(rows[self.selected].get('source_layers',[])) if self.selected is not None else ''
        initial_target=','.join(rows[self.selected].get('target_layers',[])) if self.selected is not None else ''
        source=simpledialog.askstring('修正来源地层','输入基准剖面地层编号，多个编号用逗号分隔：',initialvalue=initial_source,parent=self)
        if source is None:return
        target=simpledialog.askstring('修正目标地层','输入第二剖面地层编号，多个编号用逗号分隔：',initialvalue=initial_target,parent=self)
        if target is None:return
        source=[v.strip() for v in source.replace('，',',').split(',') if v.strip()];target=[v.strip() for v in target.replace('，',',').split(',') if v.strip()]
        a={l['id']:l for l in self.sa['layers']};b={l['id']:l for l in self.sb['layers']}
        if not source or not target or any(v not in a for v in source) or any(v not in b for v in target):messagebox.showerror('编号无效','两侧都必须填写现有地层编号。',parent=self);return
        lith={a[v]['lithology'] for v in source}|{b[v]['lithology'] for v in target}
        if len(lith)!=1 or 'UNKNOWN' in lith:messagebox.showerror('违反地质硬规则','只有经过确认且岩性完全相同的地层才能建立对应。',parent=self);return
        relation='split' if len(target)>1 else ('merge' if len(source)>1 else 'continuous')
        row={'source_layers':source,'target_layers':target,'relation_type':relation,'proposed_relation':relation,'status':'manual_review','confidence':1.,'reasons':['human_created']}
        self.custom.setdefault(self.method.get(),[]).append(row);self.decisions.setdefault(self.method.get(),{})[str(len(self.rows())-1)]='confirmed';self.refresh()

    def save_review(self):
        write(self.folder/'manual_review.json',{'schema':'stratamatch-human-review-v1','source_section':self.sa['id'],'target_section':self.sb['id'],
            'decisions':self.decisions,'custom_rows':self.custom});messagebox.showinfo('已保存','人工审查已保存，与算法原始结果分开。',parent=self)

    def export_modeling(self):
        method=self.method.get();confirmed=[];used_a=set();used_b=set()
        for i,row in enumerate(self.rows()):
            if self.decisions.get(method,{}).get(str(i))!='confirmed':continue
            sources=row.get('source_layers',[]);targets=row.get('target_layers',[])
            if not sources or not targets:continue
            if used_a.intersection(sources) or used_b.intersection(targets):messagebox.showerror('对应冲突','同一地层出现在多个已确认关系中，请先修正人工判定。',parent=self);return
            used_a.update(sources);used_b.update(targets);confirmed.append({'source_section':self.sa['id'],'source_layers':sources,'target_section':self.sb['id'],'target_layers':targets,
                'relation_type':'split' if len(targets)>1 else ('merge' if len(sources)>1 else 'continuous'),'status':'human_confirmed'})
        output=self.folder/f'modeling_input_{method}.json'
        write(output,{'schema':'stratamatch-modeling-v1','coordinate_frame':self.sa['frame'],'units':self.sa['units'],'sections':[self.sa,self.sb],
            'correspondences':confirmed,'review':{'method':method,'review_file':'manual_review.json','human_confirmed_count':len(confirmed),
                'excluded_unreviewed_count':len(self.rows())-len(confirmed)}})
        self.save_review();messagebox.showinfo('建模输入已导出',str(output),parent=self)

class Application(ttk.Frame):
    def __init__(self,root):
        super().__init__(root,padding=16);self.pack(fill='both',expand=True);self.root=root
        root.title('剖面地层匹配 · 实验工作台');root.geometry('1200x850');root.minsize(1000,740)
        style=ttk.Style(root);style.theme_use('clam');style.configure('.',font=('Microsoft YaHei UI',10));style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'));style.configure('Treeview',rowheight=28)
        self.sections=[];self.events=queue.Queue();self.worker=None;self.import_worker=None;self.cancel=threading.Event();self.last_result=None
        self.config=read('config/default.json');self.methods={m:tk.BooleanVar(value=m in ['dtw','graph','markov']) for m in METHODS}
        self.output=tk.StringVar(value=str(local('outputs/gui')));self.status=tk.StringVar(value='准备就绪：可先点击“载入演示”体验完整流程。')
        ttk.Label(self,text='剖面地层匹配',style='Title.TLabel').pack(anchor='w')
        ttk.Label(self,text='1 选择剖面与图例    →    2 选择方法及参数    →    3 运行并审查结果',foreground='#49647e').pack(anchor='w',pady=(3,12))
        notebook=ttk.Notebook(self);notebook.pack(fill='both',expand=True)
        inputs=ttk.Frame(notebook,padding=12);settings_host=ttk.Frame(notebook);results=ttk.Frame(notebook,padding=12)
        # Scroll parameters while reserving the run/output controls outside the tabs.
        scroller=tk.Canvas(settings_host,highlightthickness=0,background=style.lookup('TFrame','background'))
        scrollbar=ttk.Scrollbar(settings_host,orient='vertical',command=scroller.yview);scrollbar.pack(side='right',fill='y');scroller.pack(fill='both',expand=True)
        settings=ttk.Frame(scroller,padding=16);content=scroller.create_window(0,0,window=settings,anchor='nw')
        settings.bind('<Configure>',lambda e:scroller.configure(scrollregion=scroller.bbox('all')))
        scroller.bind('<Configure>',lambda e:scroller.itemconfigure(content,width=e.width));scroller.configure(yscrollcommand=scrollbar.set)
        notebook.add(inputs,text='  剖面与图例  ');notebook.add(settings_host,text='  方法与参数  ');notebook.add(results,text='  运行日志与结果  ');self.notebook=notebook
        root.bind('<MouseWheel>',lambda e:scroller.yview_scroll(-int(e.delta/120),'units') if notebook.select()==str(settings_host) and e.widget.winfo_toplevel()==root else None)
        buttons=ttk.Frame(inputs);buttons.pack(fill='x')
        for text,command in [('添加图片',self.add_images),('打开配置/地层 JSON',self.load_file),('保存当前配置',self.save_project),('载入演示',self.load_demo)]:ttk.Button(buttons,text=text,command=command).pack(side='left',padx=(0,8))
        self.tree=ttk.Treeview(inputs,columns=('id','station','file','legend','registration'),show='headings',height=13,selectmode='browse')
        for col,title,width in [('id','剖面编号',100),('station','空间位置',90),('file','图片 / 输入类型',380),('legend','图例 / 地层数',120),('registration','配准确认',90)]:self.tree.heading(col,text=title);self.tree.column(col,width=width)
        self.tree.pack(fill='both',expand=True,pady=12);self.tree.bind('<Double-1>',lambda e:self.edit())
        actions=ttk.Frame(inputs);actions.pack(fill='x')
        for label,cmd in [('编辑剖面 / 框选图例',self.edit),('上移',lambda:self.move(-1)),('下移',lambda:self.move(1)),('移除',self.remove)]:ttk.Button(actions,text=label,command=cmd).pack(side='left',padx=(0,8))
        ttk.Label(inputs,text='导入时自动建议主图、图例和 OCR 文字；双击剖面核对名称、位置及物理坐标。\n多主图、灰度/纹理图可能需手动定位。未确认内容不作为可靠地质匹配依据。',foreground='#49647e',wraplength=1000).pack(side='bottom',anchor='w',pady=8,before=self.tree)
        actions.pack_configure(side='bottom',before=self.tree)
        methods_frame=ttk.LabelFrame(settings,text='匹配方法（只运行选中的方法）',padding=12);methods_frame.pack(fill='x')
        for m,label in [('dtw','DTW 动态规划'),('graph','图匹配'),('markov','Markov 概率序列'),('gnn','GNN（需复核）'),('jev','Jev（需复核）'),('laya','Laya（需复核）')]:ttk.Checkbutton(methods_frame,text=label,variable=self.methods[m]).pack(side='left',padx=8)
        params=ttk.LabelFrame(settings,text='匹配与识别参数',padding=12);params.pack(fill='x',pady=12);self.params={}
        definitions=[('max_group','最大 split/merge 组大小',3),('max_candidates','匹配候选预算',1200),('large_candidate_neighbors','大实例同岩性近邻数',2),('max_accept_cost','最大接受代价（越低越严格）',.42),('min_global_margin','最小替代解间隔（越高越严格）',.28),('lab_tolerance','图例色差容许值',12),('lab_margin','图例第二候选最小间隔',4),('pale_lab_tolerance','确认白色地层色差容许值',7),('min_area_pixels','最小区域像素数',60),('min_component_fraction','大图最小区域占比',0.00005),('max_recognition_pixels','识别工作像素上限',7500000),('boundary_samples','边界采样点数',64),('contact_gap_pixels','接触间隔像素数',3),('artifact_max_gap_pixels','线状杂项修复最大跨度',32),('artifact_corridor_half_width','粗钻孔/线条最大半宽',14),('text_max_gap_pixels','文字覆盖修复最大跨度',18),('artifact_fill_passes','杂项修复迭代次数',4),('bridge_unclassified_gap_pixels','同层跨白线最大跨度',48),('internal_gap_max_fraction','内部空洞最大占比',.08),
            ('weight_position','位置软权重',.25),('weight_upper','上邻层岩性软权重',.16),('weight_lower','下邻层岩性软权重',.16),('weight_shape','形态软权重',.16),('weight_boundary','边界软权重',.08),('weight_thickness','厚度软权重',.05),('weight_area','面积软权重',.03),('weight_direction','方向软权重',.03),('weight_order','层序软权重',.04),('weight_contact','其他接触软权重',.02),('weight_topology','拓扑软权重',.02)]
        for i,(key,label,default) in enumerate(definitions):
            row,col=divmod(i,2);ttk.Label(params,text=label).grid(row=row,column=col*2,sticky='w',padx=8,pady=8)
            var=tk.StringVar(value=str(self.config.get(key,default)));self.params[key]=var;ttk.Entry(params,textvariable=var,width=14).grid(row=row,column=col*2+1,sticky='w',padx=8)
        self.artifact=tk.BooleanVar(value=bool(self.config.get('artifact_preprocessing',True)))
        ttk.Checkbutton(settings,text='识别前自动检测并保守修复网格线 / OCR 文字杂项',variable=self.artifact).pack(anchor='w')
        self.override=tk.BooleanVar(value=False);ttk.Checkbutton(settings,text='用上方识别参数覆盖各图片原有设置（默认保留已加载配置）',variable=self.override).pack(anchor='w')
        model=ttk.LabelFrame(settings,text='模型设置',padding=12);model.pack(fill='x',pady=12)
        self.model=tk.StringVar(value=self.config.get('gnn_model',''));ttk.Label(model,text='GNN 模型').grid(row=0,column=0);ttk.Entry(model,textvariable=self.model,width=70).grid(row=0,column=1,padx=8);ttk.Button(model,text='选择…',command=self.select_model).grid(row=0,column=2)
        self.jev_mode=tk.StringVar(value='mock');ttk.Label(model,text='Jev 模式').grid(row=1,column=0,pady=10);ttk.Combobox(model,textvariable=self.jev_mode,values=['mock','live'],state='readonly',width=12).grid(row=1,column=1,sticky='w',padx=8)
        self.endpoint=tk.StringVar(value=self.config['jev']['endpoint']);ttk.Label(model,text='Jev API 地址').grid(row=2,column=0);ttk.Entry(model,textvariable=self.endpoint,width=70).grid(row=2,column=1,padx=8)
        self.laya_mode=tk.StringVar(value=self.config.get('laya',{}).get('mode','mock'));ttk.Label(model,text='Laya 模式').grid(row=3,column=0,pady=10);ttk.Combobox(model,textvariable=self.laya_mode,values=['mock','http','python'],state='readonly',width=12).grid(row=3,column=1,sticky='w',padx=8)
        self.laya_endpoint=tk.StringVar(value=self.config.get('laya',{}).get('endpoint','http://127.0.0.1:8791/v1/systemone'));ttk.Label(model,text='Laya 服务地址').grid(row=4,column=0);ttk.Entry(model,textvariable=self.laya_endpoint,width=70).grid(row=4,column=1,padx=8)
        ttk.Label(model,text='Jev/Laya 使用完全相同的地质 state 与 questions。密钥只从环境变量读取；两者均强制复核。',foreground='#49647e').grid(row=5,column=0,columnspan=3,pady=10,sticky='w')
        self.log=tk.Text(results,height=12,wrap='word',font=('Microsoft YaHei UI',10),state='disabled');self.log.pack(fill='both',expand=True)
        self.result_tree=ttk.Treeview(results,columns=('page',),show='headings',height=5);self.result_tree.heading('page',text='双击打开相邻剖面对审查页面');self.result_tree.pack(fill='x',pady=8);self.result_tree.bind('<Double-1>',lambda e:self.open_result())
        rb=ttk.Frame(results);rb.pack(fill='x');ttk.Button(rb,text='打开选中结果',command=self.open_result).pack(side='left');ttk.Button(rb,text='打开已有审查文件…',command=self.open_saved_review).pack(side='left',padx=8);ttk.Button(rb,text='打开任务输出目录',command=self.open_output).pack(side='left',padx=8)
        out=ttk.LabelFrame(self,text='输出位置（遵守项目目录限制，每次运行自动创建独立子目录）',padding=10);out.pack(fill='x',pady=10)
        ttk.Entry(out,textvariable=self.output).pack(side='left',fill='x',expand=True);ttk.Button(out,text='选择目录…',command=self.choose_output).pack(side='left',padx=8)
        bar=ttk.Frame(self);bar.pack(fill='x');self.run_button=ttk.Button(bar,text='开始识别与匹配',command=self.start);self.run_button.pack(side='right');self.cancel_button=ttk.Button(bar,text='取消任务',command=self.stop,state='disabled');self.cancel_button.pack(side='right',padx=8)
        self.progress=ttk.Progressbar(bar,mode='indeterminate',length=150);self.progress.pack(side='left');ttk.Label(bar,textvariable=self.status).pack(side='left',padx=12)
        bar.pack_configure(side='bottom',before=notebook)
        out.pack_configure(side='bottom',before=notebook)
        root.protocol('WM_DELETE_WINDOW',self.close);self.poll_after=root.after(100,self.poll)

    def error(self,exc):messagebox.showerror('无法完成操作',str(exc),parent=self.root)
    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for i,s in enumerate(self.sections):
            count=str(len(s.get('layers',s.get('legend',[]))))
            if 'layers' not in s and s.get('legend_catalog'):
                count=f"{len(s.get('legend',[]))} 色块 / {s['legend_catalog'].get('cell_count',0)} 全部"
            if 'layers' not in s and any(not e.get('verified') for e in s.get('legend',[])):count+='（待确认）'
            self.tree.insert('','end',iid=str(i),values=(s['id'],s['station'],s.get('image',s.get('instance_mask','结构化地层')),count,'已确认' if s.get('registration_verified') else '未确认'))
    def selected(self):return int(self.tree.selection()[0]) if self.tree.selection() else None
    def add_images(self):
        if self.import_worker and self.import_worker.is_alive():self.error('正在导入并识别，请等待当前批次完成');return
        paths=filedialog.askopenfilenames(title='选择剖面图片',filetypes=[('剖面图片','*.png *.jpg *.jpeg *.tif *.tiff')])
        if paths:self.import_files(paths)
    def import_files(self,paths):
        def work():
            sections=[];errors=[]
            for i,p in enumerate(paths):
                self.events.put(('import_progress',f'自动定位与 OCR：{i+1}/{len(paths)}'))
                try:sections.append(prepare_image(import_image(p)))
                except Exception as exc:errors.append(f'{Path(p).name}: {exc}')
            self.events.put(('imported',(sections,errors)))
        self.import_worker=threading.Thread(target=work,daemon=True);self.import_worker.start()
    def load(self,path):
        data=read(path)
        if not isinstance(data.get('sections'),list):raise ValueError('文件必须包含 sections 数组')
        self.sections=deepcopy(data['sections']);self.refresh()
        if data.get('ui_config'):
            cfg=data['ui_config'];self.config=cfg
            for k,v in self.params.items():
                if k in cfg:v.set(str(cfg[k]))
            self.model.set(cfg.get('gnn_model',''));self.jev_mode.set(cfg.get('jev',{}).get('mode','mock'));self.endpoint.set(cfg.get('jev',{}).get('endpoint',self.endpoint.get()))
            self.laya_mode.set(cfg.get('laya',{}).get('mode','mock'));self.laya_endpoint.set(cfg.get('laya',{}).get('endpoint',self.laya_endpoint.get()))
            self.artifact.set(bool(cfg.get('artifact_preprocessing',True)))
            for m,v in self.methods.items():v.set(m in data.get('methods',METHODS))
            self.override.set(data.get('override_recognition',False))
            if data.get('output'):self.output.set(str(local(data['output'])))
        self.status.set(f'已载入 {len(self.sections)} 张剖面')
    def load_file(self):
        p=filedialog.askopenfilename(initialdir=ROOT,filetypes=[('配置或地层文件','*.json')])
        if p:
            try:self.load(p)
            except Exception as exc:self.error(exc)
    def load_demo(self):
        try:self.load('data/synthetic/images/split/manifest.json')
        except Exception as exc:self.error(exc)
    def edit(self):
        index=self.selected()
        if index is None:return
        def save(data):self.sections[index]=data;self.refresh()
        try:SectionEditor(self.root,self.sections[index],save)
        except Exception as exc:self.error(exc)
    def move(self,direction):
        i=self.selected()
        if i is not None and 0<=i+direction<len(self.sections):self.sections[i],self.sections[i+direction]=self.sections[i+direction],self.sections[i];self.refresh();self.tree.selection_set(str(i+direction))
    def remove(self):
        i=self.selected()
        if i is not None:self.sections.pop(i);self.refresh()
    def choose_output(self):
        p=filedialog.askdirectory(initialdir=local('outputs'),mustexist=False)
        if p:
            try:self.output.set(str(local(p)))
            except ValueError:self.error('按项目约束，输出目录必须位于：\n'+str(ROOT))
    def select_model(self):
        p=filedialog.askopenfilename(initialdir=local('models'),filetypes=[('GNN 权重','*.npz')])
        if p:
            try:self.model.set(str(local(p).relative_to(ROOT)))
            except Exception as exc:self.error(exc)
    def settings(self):
        cfg=deepcopy(self.config)
        integers={'max_group','max_candidates','large_candidate_neighbors','min_area_pixels','max_recognition_pixels','boundary_samples','contact_gap_pixels','artifact_max_gap_pixels','artifact_corridor_half_width','text_max_gap_pixels','artifact_fill_passes','bridge_unclassified_gap_pixels'}
        for k,v in self.params.items():cfg[k]=int(v.get()) if k in integers else float(v.get())
        import math
        if any(not math.isfinite(cfg[k]) for k in self.params):raise ValueError('参数必须为有限数字')
        if not 8<=cfg['boundary_samples']<=512 or not 1<=cfg['min_area_pixels'] or not 0<=cfg['contact_gap_pixels']<=20:raise ValueError('采样点 8–512；最小面积 ≥1；接触间隔 0–20')
        if not 0<=cfg['min_component_fraction']<=.01 or not 250000<=cfg['max_recognition_pixels']<=40000000:raise ValueError('大图最小区域占比须为 0–0.01；工作像素须为 25 万–4000 万')
        if not 100<=cfg['max_candidates']<=5000 or not 1<=cfg['large_candidate_neighbors']<=8:raise ValueError('候选预算须为 100–5000；大实例近邻数须为 1–8')
        if not 0<cfg['lab_tolerance']<=100 or not 0<cfg['lab_margin']<=100 or not 0<cfg['pale_lab_tolerance']<=30:raise ValueError('普通色差参数须为 0–100，白色地层色差须为 0–30')
        if not 1<=cfg['artifact_max_gap_pixels']<=80 or not 0<=cfg['artifact_corridor_half_width']<=40 or not 1<=cfg['text_max_gap_pixels']<=60 or not 1<=cfg['bridge_unclassified_gap_pixels']<=120 or not 1<=cfg['artifact_fill_passes']<=10 or not 0<=cfg['internal_gap_max_fraction']<=.25:raise ValueError('线状杂项跨度须为 1–80，粗钻孔半宽须为 0–40，文字跨度须为 1–60，同层跨白线跨度须为 1–120，迭代次数须为 1–10，内部空洞占比须为 0–0.25')
        weights=[cfg['weight_'+name] for name in ['position','thickness','area','shape','boundary','direction','order','upper','lower','contact','topology']]
        if any(value<0 for value in weights) or sum(weights)<=0:raise ValueError('软证据权重不得为负且不能全部为 0')
        cfg['artifact_preprocessing']=self.artifact.get()
        cfg['gnn_model']=self.model.get().strip();cfg['jev'].update(mode=self.jev_mode.get(),endpoint=self.endpoint.get().strip())
        cfg.setdefault('laya',{}).update(mode=self.laya_mode.get(),endpoint=self.laya_endpoint.get().strip())
        return cfg
    def save_project(self):
        try:
            cfg=self.settings();p=filedialog.asksaveasfilename(initialdir=local('config'),defaultextension='.json',filetypes=[('项目配置','*.json')])
            if p:write(p,{'sections':self.sections,'ui_config':cfg,'methods':[m for m,v in self.methods.items() if v.get()],'override_recognition':self.override.get(),'output':str(local(self.output.get()))})
        except Exception as exc:self.error(exc)
    def start(self):
        if self.worker and self.worker.is_alive():return
        if self.import_worker and self.import_worker.is_alive():self.error('自动导入尚未结束，请稍候');return
        try:
            cfg=self.settings();methods=[m for m,v in self.methods.items() if v.get()];sections=deepcopy(self.sections);output=self.output.get()
            if self.override.get():
                for s in sections:
                    if 'layers' not in s:
                        for k in ['lab_tolerance','lab_margin','pale_lab_tolerance','min_area_pixels','min_component_fraction','max_recognition_pixels','boundary_samples','contact_gap_pixels','artifact_max_gap_pixels','artifact_corridor_half_width','text_max_gap_pixels','artifact_fill_passes','bridge_unclassified_gap_pixels','internal_gap_max_fraction']:s[k]=cfg[k]
            validate_request(sections,methods,cfg,output)
        except Exception as exc:self.error(exc);return
        self.cancel.clear();self.run_button.configure(state='disabled');self.cancel_button.configure(state='normal');self.progress.start(15);self.notebook.select(2)
        self.status.set('运行中；输入与参数已快照，修改将在下次运行生效')
        def work():
            try:self.events.put(('done',run_job(sections,methods,cfg,output,lambda t:self.events.put(('log',t)),self.cancel)))
            except Cancelled as exc:self.events.put(('cancelled',str(exc)))
            except Exception as exc:self.events.put(('error',str(exc)))
        self.worker=threading.Thread(target=work,daemon=True);self.worker.start()
    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='import_progress':self.status.set(value)
                elif kind=='imported':
                    sections,errors=value
                    for s in sections:
                        i=len(self.sections)+1;ids={a['id'] for a in self.sections}
                        while f'S{i:02d}' in ids:i+=1
                        s['id']=f'S{i:02d}';s['station']=float(self.sections[-1]['station'])+100 if self.sections else 0
                        self.sections.append(s)
                    self.refresh();self.status.set(f'已导入 {len(sections)} 张；双击核对自动主图、图例及 OCR 名称')
                    if errors:self.error('\n'.join(errors))
                elif kind=='log':self.log.configure(state='normal');self.log.insert('end',value+'\n');self.log.see('end');self.log.configure(state='disabled')
                else:
                    self.progress.stop();self.run_button.configure(state='normal');self.cancel_button.configure(state='disabled')
                    if kind=='done':
                        self.last_result=value;self.result_tree.delete(*self.result_tree.get_children())
                        for i,p in enumerate(value['pages']):self.result_tree.insert('','end',iid=str(i),values=(p,))
                        self.status.set('完成：双击结果行查看方法对比')
                    elif kind=='cancelled':self.status.set(value)
                    else:self.status.set('运行失败：请检查输入与日志');self.error(value)
        except queue.Empty:pass
        self.poll_after=self.root.after(100,self.poll)
    def destroy(self):
        # Remove scheduled callbacks before destroying widgets (also used by tests).
        if getattr(self,'poll_after',None):
            self.root.after_cancel(self.poll_after);self.poll_after=None
        super().destroy()
    def stop(self):self.cancel.set();self.status.set('正在取消：等待当前求解阶段结束，不接受未完成结果')
    def open_result(self):
        if not self.last_result:return
        selection=self.result_tree.selection();i=int(selection[0]) if selection else 0
        reviews=self.last_result.get('reviews',[])
        if i<len(reviews) and local(reviews[i]).is_file():PairReviewWindow(self.root,reviews[i])
        elif self.last_result['pages']:os.startfile(str(local(self.last_result['pages'][i])))
    def open_saved_review(self):
        path=filedialog.askopenfilename(initialdir=local('outputs'),title='选择 review_manifest.json',filetypes=[('审查清单','review_manifest.json'),('JSON','*.json')])
        if path:
            try:PairReviewWindow(self.root,path)
            except Exception as exc:self.error(exc)
    def open_output(self):
        p=local(self.last_result['directory'] if self.last_result else self.output.get());p.mkdir(parents=True,exist_ok=True);os.startfile(str(p))
    def close(self):
        if self.import_worker and self.import_worker.is_alive():messagebox.showinfo('导入中','请等待当前自动定位/OCR 批次完成后关闭。',parent=self.root);return
        if self.worker and self.worker.is_alive():self.stop();messagebox.showinfo('任务仍在运行','已请求取消。请等当前阶段结束后关闭窗口。',parent=self.root);return
        self.root.destroy()

def main():
    configure_dpi()
    root=tk.Tk();Application(root);root.mainloop()

if __name__=='__main__':main()
