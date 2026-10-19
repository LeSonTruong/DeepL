"""Tạo notebook, bảng đối chiếu, PDF và ảnh từ lời giải/kết quả đã tính.

python build_submission.py --execute: chạy các notebook đọc cache bằng kernel
của chính Python đang chạy lệnh. Không giả lập execution_count hay output.
"""
import argparse
import asyncio
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import textwrap
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np
import pandas as pd
import torch
from data import BASE

NAME='Lê Sơn Trường'
COUNTS={
    '1.9':4,'2.1':2,'2.2':5,'2.3':12,'2.4':10,'2.5':8,'2.6':8,
    '3.1':8,'3.2':2,'3.3':4,'3.4':9,'3.5':5,'3.6':7,'3.7':6,
    '4.1':7,'4.2':3,'4.3':3,'4.4':5,'4.5':4,'4.6':4,'4.7':4,
    '5.1':7,'5.2':9,'5.3':5,'5.4':4,'5.5':5,'5.6':7,'5.7':5,
}
PAGES={'1.9':69,'2.1':77,'2.2':80,'2.3':93,'2.4':99,'2.5':104,'2.6':117,
       '3.1':132,'3.2':138,'3.3':142,'3.4':147,'3.5':151,'3.6':157,'3.7':164,
       '4.1':172,'4.2':177,'4.3':179,'4.4':184,'4.5':187,'4.6':194,'4.7':205,
       '5.1':215,'5.2':219,'5.3':223,'5.4':228,'5.5':233,'5.6':238,'5.7':246}
CHAPTER_URLS={1:'chapter_introduction/index.html',2:'chapter_preliminaries/index.html',
              3:'chapter_linear-regression/index.html',4:'chapter_linear-classification/index.html',
              5:'chapter_multilayer-perceptrons/index.html'}

SETUP='''from pathlib import Path
import sys

cwd = Path.cwd().resolve()
paths = [cwd, cwd / "bai_lam_Le_Son_Truong", cwd / "Buoi2" / "bai_lam_Le_Son_Truong",
         *[p / "Buoi2" / "bai_lam_Le_Son_Truong" for p in cwd.parents]]
BASE = next((p for p in paths if (p / "experiments.py").is_file()), None)
if BASE is None:
    raise FileNotFoundError("Mở notebook trong repository DeepL hoặc thư mục bài làm Buoi2.")
sys.path.insert(0, str(BASE))
from experiments import run_section, render_result

# False: đọc lại kết quả thật đã lưu. True: chạy lại các thí nghiệm (tốn thời gian).
RECOMPUTE = False
if RECOMPUTE:
    from data import prepare
    prepare()
print("Người làm: Lê Sơn Trường | D2L 1.0.3 | CPU, seed 42")
'''


def section_blocks(text):
    matches=list(re.finditer(r'^## (\d+\.\d+) —.*$',text,re.M))
    return [(m.group(1),text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)].strip())
            for i,m in enumerate(matches)]


def quantitative_notes(section):
    path=BASE/'results'/section.replace('.','_')
    if section=='3.7':
        df=pd.read_csv(path/'weight_decay.csv');best=df.loc[df.validation_mse.idxmin()]
        return f"**Kết quả SGD đúng dữ liệu mục 3.7:** bias thật 0.05, 20 train/100 validation, batch5, LR=0.01, 10 epoch. λ={best['lambda']:g} có validation MSE thấp nhất lưới ({best.validation_mse:.6f}); λ=0 cho {df.iloc[0].validation_mse:.6f}. Bảng exact ridge là tham chiếu riêng."
    if section=='4.4':
        df=pd.read_csv(path/'comparison.csv');best=df.loc[df.val_loss.idxmin()]
        return f"**Nhận xét từ số đo:** {best['name']} có validation loss thấp nhất trong lưới softmax ({best.val_loss:.4f}), accuracy {best.val_accuracy:.2%}. Batch nhỏ tạo nhiều update hơn mỗi epoch, nên không coi đây là so sánh cùng số bước."
    if section=='5.2':
        df=pd.read_csv(path/'comparison.csv');best=df.loc[df.val_loss.idxmin()]
        widths=df[df.name.str.startswith('width_')];wide=widths.loc[widths.val_accuracy.idxmax()]
        return f"**Nhận xét từ số đo:** trong nhóm độ rộng, {wide['name']} có accuracy cao nhất ({wide.val_accuracy:.2%}). Trong toàn bộ cấu hình thử, {best['name']} có loss thấp nhất ({best.val_loss:.4f}), accuracy {best.val_accuracy:.2%}. Zero init và bottleneck 1 neuron chỉ đạt 10% trong lần chạy này. Không suy rộng thành siêu tham số tối ưu toàn cục."
    if section=='5.6':
        df=pd.read_csv(path/'comparison.csv').set_index('name')
        return (f"**Nhận xét từ số đo tại 20 epoch:** không dropout {df.loc['no_dropout','val_accuracy']:.2%}; "
                f"p=(0.2,0.5) {df.loc['dropout_02_05','val_accuracy']:.2%}; p=(0.5,0.2) {df.loc['dropout_05_02','val_accuracy']:.2%}; "
                f"weight decay {df.loc['weight_decay','val_accuracy']:.2%}; kết hợp {df.loc['dropout_plus_decay','val_accuracy']:.2%}; "
                f"Gaussian noise {df.loc['gaussian_noise','val_accuracy']:.2%}. Các mức regularization này chưa vượt baseline không dropout; không bịa kết luận cải thiện.")
    if section=='5.7':
        df=pd.read_csv(path/'model_selection.csv');best=df.loc[df.cv_log_rmse_mean.idxmin()]
        return f"**Nhận xét từ số đo:** {best['name']} có 5-fold log RMSE {best.cv_log_rmse_mean:.5f} ± {best.cv_log_rmse_std:.5f}; baseline 10 epoch là {df.iloc[0].cv_log_rmse_mean:.5f}. Đây là CV local, **chưa phải điểm Kaggle**. CSV dự đoán đã tạo, còn thao tác nộp bằng tài khoản cá nhân."
    return ''


def build_notebook(chapter,execute):
    text=(BASE/'loi_giai'/f'chuong{chapter}.md').read_text(encoding='utf-8')
    intro=text.split('\n## ',1)[0]
    cells=[nbf.v4.new_markdown_cell(intro+f'\n\nNguồn: [D2L chương {chapter}](https://d2l.ai/{CHAPTER_URLS[chapter]}). '
           'Lời giải giữ số câu để đối chiếu PDF. Chạy **Run All**; mặc định đọc cache kết quả thật. '
           'Đổi `RECOMPUTE=True` để học lại bằng nguồn `.py` đi kèm.'),nbf.v4.new_code_cell(SETUP)]
    inventory=json.loads((BASE/'source_reference'/'exercises_inventory.json').read_text(encoding='utf-8'))
    source_by_page={item['pdf_page']:item for item in inventory}
    for section,block in section_blocks(text):
        original=source_by_page.get(PAGES.get(section))
        if original:
            heading,body=block.split('\n',1)
            source_card=(f'\n\n<details><summary>Đối chiếu đề gốc — PDF trang {PAGES[section]} (bấm để mở)</summary>\n\n'
                         'Bản trích text có thể mất vị trí chỉ số/số mũ; xem PDF gốc để đọc công thức chính xác.\n\n'
                         f'<pre>{html.escape(original["text"])}</pre>\n</details>\n\n')
            block=heading+source_card+body
        cells.append(nbf.v4.new_markdown_cell(block))
        if section.startswith('1.'):
            cells.append(nbf.v4.new_code_cell(f'print("Chương 1: đã trình bày đủ {COUNTS[section]} câu thảo luận; không có thí nghiệm số bắt buộc.")'))
        else:
            cells.append(nbf.v4.new_code_cell(f'result = run_section("{section}", recompute=RECOMPUTE)\nrender_result(result)'))
            notes=quantitative_notes(section)
            if notes:cells.append(nbf.v4.new_markdown_cell(notes))
    if chapter==5:
        cells.append(nbf.v4.new_markdown_cell('## Phần cần thao tác cá nhân\n\n'
            '5.7 câu 1: nộp CSV lên Kaggle và thêm ảnh điểm thật. 5.2 câu 7.1: benchmark GPU '
            'chưa đo vì kernel CPU; có `benchmark_gpu.py` để chạy bằng môi trường CUDA. '
            'Ảnh/PDF bài làm chưa được upload lên Drive.'))
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'name':'python3','display_name':'Python 3 (PyTorch)','language':'python'},
        'language_info':{'name':'python','version':sys.version.split()[0]},
    })
    path=BASE/f'Le_Son_Truong_Chuong_{chapter}.ipynb'
    if not execute and path.is_file():
        previous=nbf.read(path,as_version=4)
        old_code=[c for c in previous.cells if c.cell_type=='code']
        new_code=[c for c in nb.cells if c.cell_type=='code']
        if [c.source for c in old_code]!=[c.source for c in new_code]:
            raise RuntimeError('Notebook code changed: rebuild with --execute to refresh outputs.')
        for new,old in zip(new_code,old_code):
            new.outputs=old.outputs
            new.execution_count=old.execution_count
    if execute:
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        try:
            print(f'Executing notebook chapter {chapter} (cached results)...',flush=True)
            NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(BASE)}}).execute()
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbf.validate(nb);nbf.write(nb,path)


def coverage():
    rows=[]
    for chapter in range(1,6):
        text=(BASE/'loi_giai'/f'chuong{chapter}.md').read_text(encoding='utf-8')
        for section,block in section_blocks(text):
            actual=[int(n) for n in re.findall(r'^### Câu (\d+)',block,re.M)]
            expected=COUNTS.get(section,0)
            assert actual==list(range(1,expected+1)),(section,actual,expected)
            status='answer_present; theory reviewed; see KIEM_TRA_LAI.md'
            if section=='2.7':status='no_numbered_exercises'
            if section=='5.2':status='partial: answers and CPU results present; GPU timing pending CUDA'
            if section=='5.7':status='partial: local CV and CSV ready; Kaggle upload/leaderboard pending user account'
            rows.append({'chapter':chapter,'section':section,'pdf_page':PAGES.get(section,118),
                         'questions':expected,'status':status,
                         'notebook':f'Le_Son_Truong_Chuong_{chapter}.ipynb'})
    assert sum(row['questions'] for row in rows)==sum(COUNTS.values())
    pd.DataFrame(rows).to_csv(BASE/'coverage.csv',index=False,encoding='utf-8-sig')
    return rows


def clean_lines(text,width=100):
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',text)
    text=text.replace('**','').replace('`','')
    lines=[]
    for line in text.splitlines():
        line=re.sub(r'^#+\s*','',line)
        lines.extend(textwrap.wrap(line,width=width,replace_whitespace=False,break_long_words=True) or [''])
    return lines


def export_pages(rows):
    folder=BASE/'anh_nop_bai';folder.mkdir(exist_ok=True)
    # Chỉ thay ảnh có tên do chương trình này sinh; tránh giữ trang cũ sau khi số trang đổi.
    for old in folder.glob('Le_Son_Truong_[0-9][0-9][0-9]_*.png'):
        old.unlink()
    page_number=0
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    with PdfPages(BASE/'Le_Son_Truong_Buoi2.pdf') as pdf:
        def save(fig,slug):
            nonlocal page_number
            page_number+=1
            fig.text(.5,.018,f'{NAME} — Buổi 2 — Trang {page_number}',ha='center',fontsize=9)
            pdf.savefig(fig)
            fig.savefig(folder/f'Le_Son_Truong_{page_number:03}_{slug}.png',dpi=140)
            plt.close(fig)
        def text_pages(title,text,slug):
            lines=clean_lines(text)
            for start in range(0,len(lines),57):
                fig=plt.figure(figsize=(8.27,11.69))
                fig.text(.065,.965,title,fontsize=13,weight='bold',va='top')
                fig.text(.065,.922,'\n'.join(lines[start:start+57]),fontsize=9,linespacing=1.35,va='top')
                save(fig,slug)
        text_pages('BÀI TẬP BUỔI 2 — LÊ SƠN TRƯỜNG',
            'D2L 1.0.3 — Exercises chương 1–5\n\n'
            f'{sum(COUNTS.values())} câu cấp chính, 28 nhóm Exercises; có lời giải các ý con.\n'
            'Đã chạy thí nghiệm bằng PyTorch, seed42, và lưu output notebook.\n\n'
            'Phần còn cần thao tác: nộp CSV Kaggle để có điểm leaderboard; '
            'đo GPU bằng môi trường CUDA nếu bắt buộc; upload hồ sơ lên Drive.\n'
            'Không giả lập các ảnh màn hình tài khoản. Đây là bản xuất lời giải và kết quả thực nghiệm.\n\n'
            'Dữ liệu: Abalone UCI, Fashion-MNIST chính thức, House Prices của D2L.\n'
            'Xem README.md để chạy lại, coverage.csv để đối chiếu từng mục.','gioi_thieu')
        for chapter in range(1,6):
            text=(BASE/'loi_giai'/f'chuong{chapter}.md').read_text(encoding='utf-8')
            for section,block in section_blocks(text):
                text_pages(f'{NAME} — Chương {chapter}, mục {section}',block,f'c{chapter}_{section.replace(".","_")}_loi_giai')
                result_path=BASE/'results'/section.replace('.','_')/'result.json'
                if not result_path.is_file():continue
                result=json.loads(result_path.read_text(encoding='utf-8'))
                if result['notes']:
                    text_pages(f'Ghi nhận chạy thật — mục {section}',
                               '\n\n'.join(result['notes']),f'{section}_output')
                notes=quantitative_notes(section)
                if notes:text_pages(f'Kết luận thực nghiệm — mục {section}',notes,f'{section}_ket_luan')
                for table_path in result['tables']:
                    if Path(table_path).stem=='history':continue
                    frame=pd.read_csv(BASE/table_path)
                    if len(frame.columns)>9:
                        selected=[c for c in ['name','hidden','epoch','lr','train_loss','val_loss','train_accuracy','val_accuracy','cv_log_rmse_mean','cv_log_rmse_std'] if c in frame]
                        if selected:frame=frame[selected]
                    text_pages(f'Kết quả mục {section} — {Path(table_path).stem}',
                               frame.head(30).to_string(index=False,float_format=lambda x:f'{x:.6g}'),f'{section}_{Path(table_path).stem}')
                for image_path in result['figures']:
                    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.imshow(plt.imread(BASE/image_path));ax.axis('off')
                    fig.suptitle(f'{NAME} — Kết quả mục {section}\n{Path(image_path).stem}',fontsize=13)
                    fig.subplots_adjust(left=.03,right=.97,top=.9,bottom=.07)
                    save(fig,f'{section}_{Path(image_path).stem}')
        text_pages(f'{NAME} — Ôn tập Loss, backpropagation và MLP',
                   (BASE/'ON_TAP_LOSS_BACKPROP_MLP.md').read_text(encoding='utf-8'),'on_tap')
    print(f'Exported PDF and {page_number} result/answer images.',flush=True)


def validate_artifacts():
    rows=pd.read_csv(BASE/'coverage.csv');assert rows.questions.sum()==sum(COUNTS.values())
    for section in list(COUNTS)[1:]+['2.7']:
        result_path=BASE/'results'/section.replace('.','_')/'result.json'
        assert result_path.is_file(),section
        result=json.loads(result_path.read_text(encoding='utf-8'))
        for item in result['tables']+result['figures']:assert (BASE/item).is_file(),item
    submission=pd.read_csv(BASE/'Le_Son_Truong_submission.csv')
    test=pd.read_csv(BASE/'data'/'kaggle_house_pred_test.csv')
    assert submission.columns.tolist()==['Id','SalePrice'] and len(submission)==1459
    assert np.array_equal(submission.Id.values,test.Id.values)
    assert np.isfinite(submission.SalePrice).all() and (submission.SalePrice>0).all()
    for chapter in range(1,6):
        nb=nbf.read(BASE/f'Le_Son_Truong_Chuong_{chapter}.ipynb',as_version=4);nbf.validate(nb)
        assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
        assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    print(f'Validated {sum(COUNTS.values())} numbered answers, result files, five executed notebooks and Kaggle CSV schema.',flush=True)


def package():
    version={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
             'matplotlib':matplotlib.__version__,'seed':42,'threads':2,'device':'cpu',
             'cuda_available':torch.cuda.is_available(),
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}}
    (BASE/'run_metadata.json').write_text(json.dumps(version,indent=2),encoding='utf-8')
    target=BASE.parent/'Le_Son_Truong_Buoi2.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in BASE.rglob('*'):
            relative=path.relative_to(BASE)
            if not path.is_file() or any(part in ['__pycache__','data','.ipynb_checkpoints'] for part in relative.parts):continue
            if path.suffix=='.whl':continue
            archive.write(path,arcname=Path('Le_Son_Truong_Buoi2')/relative)
    print(f'Package: {target.name} ({target.stat().st_size/1024**2:.1f} MiB)',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    rows=coverage()
    for chapter in range(1,6):build_notebook(chapter,args.execute)
    validate_artifacts()
    export_pages(rows)
    from merge_submission_images import main as merge_images
    merge_images()
    package()
