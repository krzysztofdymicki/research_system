import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import shutil
from pathlib import Path

from .orchestrator import (
    SearchOptions,
    run_search_and_save,
    analyze_with_progress,
    promote_kept,
)
from .db import init_db, list_raw_results, list_publications, reset_db, update_publication_assets
from .models import Publication
from .sources.source import download_pdf_for_publication, extract_text_from_pdf


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Research System - Simple GUI")
        self.geometry("640x480")

        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Search tab
        frm = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frm, text="Search")

        ttk.Label(frm, text="Query:").grid(row=0, column=0, sticky=tk.W)
        self.query_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.query_var, width=60).grid(row=0, column=1, columnspan=3, sticky=tk.W)

        ttk.Label(frm, text="Max results per provider:").grid(row=1, column=0, sticky=tk.W)
        self.max_var = tk.IntVar(value=5)
        ttk.Spinbox(frm, from_=1, to=50, textvariable=self.max_var, width=6).grid(row=1, column=1, sticky=tk.W)

        self.use_arxiv = tk.BooleanVar(value=True)
        self.use_core = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Use arXiv", variable=self.use_arxiv).grid(row=2, column=0, sticky=tk.W)
        ttk.Checkbutton(frm, text="Use CORE", variable=self.use_core).grid(row=2, column=1, sticky=tk.W)

        # arXiv scope: Title / Abstract (at least one when arXiv enabled)
        ttk.Label(frm, text="arXiv fields:").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        self.arxiv_in_title = tk.BooleanVar(value=True)
        self.arxiv_in_abstract = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Title", variable=self.arxiv_in_title).grid(row=3, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(frm, text="Abstract", variable=self.arxiv_in_abstract).grid(row=3, column=2, sticky=tk.W, pady=(6, 0))

        # Search runs metadata-only; downloads happen later for kept
        self.run_btn = ttk.Button(frm, text="Run Search", command=self.on_run)
        self.run_btn.grid(row=4, column=0, pady=10, sticky=tk.W)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).grid(row=5, column=0, columnspan=4, sticky=tk.W)

        # Live query preview
        ttk.Separator(frm).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        ttk.Label(frm, text="Effective arXiv query:").grid(row=7, column=0, sticky=tk.W)
        self.arxiv_preview_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.arxiv_preview_var, wraplength=520).grid(row=7, column=1, columnspan=3, sticky=tk.W)
        ttk.Label(frm, text="Effective CORE query:").grid(row=8, column=0, sticky=tk.W)
        self.core_preview_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.core_preview_var, wraplength=520).grid(row=8, column=1, columnspan=3, sticky=tk.W)

        ttk.Separator(frm).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        ttk.Label(frm, text="Reset:").grid(row=10, column=0, sticky=tk.W)
        ttk.Button(frm, text="Reset DB + papers now", command=self.on_reset_now).grid(row=10, column=1, sticky=tk.W)

        for i in range(4):
            frm.columnconfigure(i, weight=1)

        # Bind preview updates
        self.query_var.trace_add('write', lambda *_: self.update_query_preview())
        self.use_arxiv.trace_add('write', lambda *_: self.update_query_preview())
        self.use_core.trace_add('write', lambda *_: self.update_query_preview())
        self.arxiv_in_title.trace_add('write', lambda *_: self.update_query_preview())
        self.arxiv_in_abstract.trace_add('write', lambda *_: self.update_query_preview())
        self.update_query_preview()

        # Search Results tab
        res = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(res, text="Search Results")

        self.tree = ttk.Treeview(
            res,
            columns=("title", "source", "query", "relevance", "analyzed"),
            show="headings",
            height=12,
        )
        self.tree.heading("title", text="Title")
        self.tree.heading("source", text="Source")
        self.tree.heading("query", text="Query")
        self.tree.heading("relevance", text="Score")
        self.tree.heading("analyzed", text="Analyzed")
        self.tree.column("title", width=360)
        self.tree.column("source", width=80)
        self.tree.column("query", width=180)
        self.tree.column("relevance", width=60, anchor=tk.E)
        self.tree.column("analyzed", width=80)
        self.tree.grid(row=0, column=0, sticky="nsew", pady=(8, 8))
        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)

        # Analysis controls
        ctrl = ttk.Frame(res)
        ctrl.grid(row=1, column=0, sticky="ew")
        ttk.Label(ctrl, text="Research title:").pack(side=tk.LEFT)
        self.research_title_var = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self.research_title_var, width=36).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(ctrl, text="AI threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=70)
        ttk.Spinbox(ctrl, from_=0, to=100, textvariable=self.threshold_var, width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(ctrl, text="Analyze Pending", command=self.on_run_ai).pack(side=tk.LEFT)
        # Progress and cancel controls
        self.ai_progress = ttk.Progressbar(ctrl, length=160, mode="determinate", maximum=100)
        self.ai_progress.pack(side=tk.LEFT, padx=(10, 6))
        self.ai_progress_label = ttk.Label(ctrl, text="0/0")
        self.ai_progress_label.pack(side=tk.LEFT)
        self.ai_cancel_requested = False
        self.ai_cancel_btn = ttk.Button(ctrl, text="Cancel", command=self.on_cancel_ai)
        self.ai_cancel_btn.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(ctrl, text="Promote to Publications", command=self.on_promote).pack(side=tk.LEFT, padx=(10, 0))

        self.detail = tk.Text(res, height=8, wrap=tk.WORD)
        self.detail.grid(row=2, column=0, sticky="nsew")

        res.rowconfigure(0, weight=1)
        res.rowconfigure(2, weight=1)
        res.columnconfigure(0, weight=1)

        # Publications tab
        pubs = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(pubs, text="Publications")

        self.download_btn = ttk.Button(pubs, text="Download PDFs", command=self.on_download_pubs)
        self.download_btn.grid(row=0, column=0, sticky=tk.W)
        self.extract_btn = ttk.Button(pubs, text="Extract Markdown", command=self.on_extract_pubs)
        self.extract_btn.grid(row=0, column=1, sticky=tk.W, padx=6)

        self.pubs_tree = ttk.Treeview(
            pubs,
            columns=("title", "source", "pdf", "md"),
            show="headings",
            height=12,
        )
        self.pubs_tree.heading("title", text="Title")
        self.pubs_tree.heading("source", text="Source")
        self.pubs_tree.heading("pdf", text="PDF")
        self.pubs_tree.heading("md", text="Markdown")
        self.pubs_tree.column("title", width=420)
        self.pubs_tree.column("source", width=80)
        self.pubs_tree.column("pdf", width=120)
        self.pubs_tree.column("md", width=100)
        self.pubs_tree.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 8))

        pubs.rowconfigure(1, weight=1)
        pubs.columnconfigure(0, weight=1)
        pubs.columnconfigure(1, weight=0)
        pubs.columnconfigure(2, weight=0)

        # Initial loads
        self._rows_by_id = {}
        self.refresh_results()
        self.refresh_publications()

    def on_run(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Validation", "Please enter a query.")
            return
        # Validate arXiv field selection
        if self.use_arxiv.get() and not (self.arxiv_in_title.get() or self.arxiv_in_abstract.get()):
            messagebox.showwarning("Validation", "For arXiv, select at least one field: Title or Abstract.")
            return

        opts = SearchOptions(
            query=query,
            max_results=self.max_var.get(),
            use_arxiv=self.use_arxiv.get(),
            use_core=self.use_core.get(),
            arxiv_in_title=self.arxiv_in_title.get(),
            arxiv_in_abstract=self.arxiv_in_abstract.get(),
            download_pdfs=False,
            extract_markdown=False,
        )

        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Running search ...")

        def work():
            try:
                total, saved = run_search_and_save(opts)
                self.status_var.set(f"Search complete. Found {total}, saved {saved} unique.")
                # Auto-switch to Search Results
                self.after(0, lambda: self.notebook.select(1))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Search Error", str(e)))
            finally:
                self.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.after(0, self.refresh_results)

        threading.Thread(target=work, daemon=True).start()

    def update_query_preview(self):
        try:
            from .sources.arxiv_source import ArxivSource
            q = self.query_var.get() or ""
            # arXiv preview only if enabled
            if self.use_arxiv.get():
                self.arxiv_preview_var.set(
                    ArxivSource.build_arxiv_query(
                        q,
                        in_title=self.arxiv_in_title.get(),
                        in_abstract=self.arxiv_in_abstract.get(),
                    )
                )
            else:
                self.arxiv_preview_var.set("(arXiv disabled)")
            # CORE preview: quotes stripped
            if self.use_core.get():
                self.core_preview_var.set(q.replace('"', '').replace("'", ''))
            else:
                self.core_preview_var.set("(CORE disabled)")
        except Exception:
            # Keep UI resilient even if import fails
            pass

    def refresh_results(self):
        conn = init_db()
        rows = list_raw_results(conn, limit=500)
        self._rows_by_id = {}
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in rows:
            rid = str(r["id"])
            score = r.get("relevance_score")
            analyzed = "yes" if score is not None else "no"
            score_str = "" if score is None else str(int(score)) if isinstance(score, (int, float)) else str(score)
            self.tree.insert("", tk.END, iid=rid, values=(r["title"], r["source"], r.get("query") or "", score_str, analyzed))
            self._rows_by_id[rid] = r

    def on_row_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        rid = sel[0]
        r = self._rows_by_id.get(rid)
        if not r:
            return
        authors = ", ".join(r.get("authors") or [])
        lines = [
            f"Title: {r['title']}",
            f"Source: {r['source']}",
            f"Query: {r.get('query') or ''}",
            f"URL: {r['url']}",
            f"PDF (raw): {r.get('pdf_url')}",
            f"Relevance: {r.get('relevance_score')}",
            f"Authors: {authors}",
            f"Abstract: {r.get('abstract') or ''}",
        ]
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", "\n".join(lines))

    def on_run_ai(self):
        threshold = self.threshold_var.get()
        self.ai_cancel_requested = False

        def cancel_flag() -> bool:
            return self.ai_cancel_requested

        def progress_cb(done: int, total: int, kept: int):
            try:
                pct = int((done / total) * 100) if total else 0
            except Exception:
                pct = 0
            def update_ui():
                self.ai_progress['value'] = pct
                self.ai_progress_label.config(text=f"{done}/{total} ({kept} kept)")
                try:
                    self.refresh_results()
                except Exception:
                    pass
            try:
                self.after(0, update_ui)
            except Exception:
                pass

        def work():
            try:
                analyzed, kept = analyze_with_progress(
                    None,
                    threshold,
                    cancel_flag,
                    progress_cb,
                    self.research_title_var.get().strip() or None,
                )
                self.after(0, lambda: self.status_var.set(f"AI analyzed {analyzed}, >= {threshold}: {kept}."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("AI Error", str(e)))
            finally:
                try:
                    self.after(0, lambda: self.ai_progress.config(value=0))
                    self.after(0, lambda: self.ai_progress_label.config(text="0/0"))
                except Exception:
                    pass
                self.after(0, self.refresh_results)

        threading.Thread(target=work, daemon=True).start()

    def on_cancel_ai(self):
        self.ai_cancel_requested = True

    def on_promote(self):
        threshold = self.threshold_var.get()

        def work():
            try:
                inserted = promote_kept(threshold, None)
                self.after(0, lambda: self.status_var.set(f"Promoted {inserted} items to Publications."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Promote Error", str(e)))
            finally:
                self.after(0, self.refresh_publications)

        threading.Thread(target=work, daemon=True).start()

    def on_download_pubs(self):
        self.download_btn.config(state=tk.DISABLED)
        self.extract_btn.config(state=tk.DISABLED)
        
        def work():
            try:
                # Get list of publications needing PDFs
                conn = init_db()
                pubs = list_publications(conn, limit=500)
                to_download = [p for p in pubs if not p.get("pdf_path")]
                total = len(to_download)
                downloaded = 0
                
                for idx, pub in enumerate(to_download, 1):
                    try:
                        # Update status before each download
                        self.after(0, lambda i=idx, t=total: self.status_var.set(f"Downloading PDF {i}/{t}..."))
                        
                        # Convert dict to Publication model
                        pub_obj = Publication(
                            id=pub["id"],
                            original_id=pub.get("original_id"),
                            title=pub["title"],
                            authors=pub.get("authors") or [],
                            abstract=pub.get("abstract"),
                            url=pub["url"],
                            pdf_url=pub.get("pdf_url"),
                            source=pub["source"],
                        )
                        
                        pdf_path = download_pdf_for_publication(pub_obj)
                        if pdf_path:
                            conn = init_db()
                            update_publication_assets(conn, publication_id=pub["id"], pdf_path=pdf_path)
                            downloaded += 1
                            
                        # Refresh GUI after each successful download
                        self.after(0, self.refresh_publications)
                        
                    except Exception:
                        pass  # Continue with next PDF
                
                self.after(0, lambda: self.status_var.set(f"PDFs: attempted {total}, downloaded {downloaded}."))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Download Error", str(e)))
            finally:
                self.after(0, lambda: self.download_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.extract_btn.config(state=tk.NORMAL))
                self.after(0, self.refresh_publications)
        
        threading.Thread(target=work, daemon=True).start()

    def on_extract_pubs(self):
        self.download_btn.config(state=tk.DISABLED)
        self.extract_btn.config(state=tk.DISABLED)
        
        def work():
            try:
                # Get list of publications with PDFs but no markdown
                conn = init_db()
                pubs = list_publications(conn, limit=500)
                to_extract = [p for p in pubs if p.get("pdf_path") and not p.get("markdown")]
                total = len(to_extract)
                extracted = 0
                
                for idx, pub in enumerate(to_extract, 1):
                    try:
                        # Update status before each extraction
                        self.after(0, lambda i=idx, t=total: self.status_var.set(f"Extracting markdown {i}/{t}..."))
                        
                        text = extract_text_from_pdf(pub["pdf_path"])
                        if text:
                            conn = init_db()
                            update_publication_assets(conn, publication_id=pub["id"], markdown=text)
                            extracted += 1
                            
                        # Refresh GUI after each successful extraction
                        self.after(0, self.refresh_publications)
                        
                    except Exception:
                        pass  # Continue with next extraction
                
                self.after(0, lambda: self.status_var.set(f"Markdown: attempted {total}, extracted {extracted}."))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Extract Error", str(e)))
            finally:
                self.after(0, lambda: self.download_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.extract_btn.config(state=tk.NORMAL))
                self.after(0, self.refresh_publications)
        
        threading.Thread(target=work, daemon=True).start()

    def refresh_publications(self):
        conn = init_db()
        rows = list_publications(conn, limit=200)
        for item in self.pubs_tree.get_children():
            self.pubs_tree.delete(item)
        for r in rows:
            pdf = "yes" if r.get("pdf_path") else "no"
            md = "yes" if r.get("markdown") else "no"
            self.pubs_tree.insert("", tk.END, iid=str(r["id"]), values=(r["title"], r["source"], pdf, md))

    # Reset helpers
    def _reset_database_and_papers(self):
        try:
            reset_db(os.path.join(self.root_dir, "research.db"))
        except Exception:
            pass
        papers_dir = Path(self.root_dir) / "papers"
        if papers_dir.is_dir():
            for path in papers_dir.iterdir():
                try:
                    if path.is_dir():
                        shutil.rmtree(str(path), ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                except Exception:
                    pass

    def on_reset_now(self):
        if messagebox.askyesno("Confirm Reset", "This will delete research.db and all files in papers/. Continue?"):
            self._reset_database_and_papers()
            self.status_var.set("Reset completed.")
            self.refresh_results()
            self.refresh_publications()


if __name__ == "__main__":
    app = App()
    app.mainloop()
