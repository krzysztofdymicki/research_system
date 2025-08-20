import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import shutil

from .orchestrator import (
    SearchOptions,
    run_search_and_save,
    analyze_with_progress,
    promote_kept,
    download_pdfs_for_publications,
    extract_markdown_for_publications,
)
from .db import init_db, list_raw_results, list_publications, reset_db


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Research System - Simple GUI")
        self.geometry("640x480")

        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._maybe_reset_on_start()

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

        # Search runs metadata-only; downloads happen later for kept
        self.run_btn = ttk.Button(frm, text="Run Search", command=self.on_run)
        self.run_btn.grid(row=3, column=0, pady=10, sticky=tk.W)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).grid(row=4, column=0, columnspan=4, sticky=tk.W)

        ttk.Separator(frm).grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        ttk.Label(frm, text="Reset:").grid(row=6, column=0, sticky=tk.W)
        ttk.Button(frm, text="Reset DB + papers now", command=self.on_reset_now).grid(row=6, column=1, sticky=tk.W)
        self.reset_on_start_var = tk.BooleanVar(value=self._is_reset_marker_present())
        ttk.Checkbutton(frm, text="Reset on next start", variable=self.reset_on_start_var, command=self.on_toggle_reset_on_start).grid(row=6, column=2, sticky=tk.W)

        for i in range(4):
            frm.columnconfigure(i, weight=1)

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

        ttk.Button(pubs, text="Download PDFs", command=self.on_download_pubs).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(pubs, text="Extract Markdown", command=self.on_extract_pubs).grid(row=0, column=1, sticky=tk.W, padx=6)

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

        opts = SearchOptions(
            query=query,
            max_results=self.max_var.get(),
            use_arxiv=self.use_arxiv.get(),
            use_core=self.use_core.get(),
            download_pdfs=False,
            extract_markdown=False,
        )

        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Running search ...")

        def work():
            try:
                total, saved = run_search_and_save(opts)
                self.status_var.set(f"Search complete. Found {total}, saved {saved} unique.")
            except Exception as e:
                messagebox.showerror("Search Error", str(e))
            finally:
                self.run_btn.config(state=tk.NORMAL)
                self.refresh_results()

        threading.Thread(target=work, daemon=True).start()

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
                analyzed, kept = analyze_with_progress(None, threshold, cancel_flag, progress_cb)
                self.status_var.set(f"AI analyzed {analyzed}, >= {threshold}: {kept}.")
            except Exception as e:
                messagebox.showerror("AI Error", str(e))
            finally:
                try:
                    self.after(0, lambda: self.ai_progress.config(value=0))
                    self.after(0, lambda: self.ai_progress_label.config(text="0/0"))
                except Exception:
                    pass
                self.refresh_results()

        threading.Thread(target=work, daemon=True).start()

    def on_cancel_ai(self):
        self.ai_cancel_requested = True

    def on_promote(self):
        threshold = self.threshold_var.get()

        def work():
            try:
                inserted = promote_kept(threshold, None)
                self.status_var.set(f"Promoted {inserted} items to Publications.")
            except Exception as e:
                messagebox.showerror("Promote Error", str(e))
            finally:
                self.refresh_publications()

        threading.Thread(target=work, daemon=True).start()

    def on_download_pubs(self):
        def work():
            try:
                attempted, downloaded = download_pdfs_for_publications()
                self.status_var.set(f"PDFs: attempted {attempted}, downloaded {downloaded}.")
            except Exception as e:
                messagebox.showerror("Download Error", str(e))
            finally:
                self.refresh_publications()
        threading.Thread(target=work, daemon=True).start()

    def on_extract_pubs(self):
        def work():
            try:
                attempted, extracted = extract_markdown_for_publications()
                self.status_var.set(f"Markdown: attempted {attempted}, extracted {extracted}.")
            except Exception as e:
                messagebox.showerror("Extract Error", str(e))
            finally:
                self.refresh_publications()
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
        papers_dir = os.path.join(self.root_dir, "papers")
        if os.path.isdir(papers_dir):
            for name in os.listdir(papers_dir):
                path = os.path.join(papers_dir, name)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                except Exception:
                    pass

    def _reset_marker_path(self) -> str:
        return os.path.join(self.root_dir, ".reset_on_start")

    def _is_reset_marker_present(self) -> bool:
        return os.path.exists(self._reset_marker_path())

    def _maybe_reset_on_start(self):
        env_flag = os.environ.get("RS_RESET_ON_START", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        marker = os.path.exists(os.path.join(self.root_dir, ".reset_on_start"))
        if env_flag or marker:
            self._reset_database_and_papers()
            if marker:
                try:
                    os.remove(self._reset_marker_path())
                except Exception:
                    pass
            try:
                messagebox.showinfo("Reset", "Database and papers directory have been reset.")
            except Exception:
                pass

    def on_reset_now(self):
        if messagebox.askyesno("Confirm Reset", "This will delete research.db and all files in papers/. Continue?"):
            self._reset_database_and_papers()
            self.status_var.set("Reset completed.")
            self.refresh_results()

    def on_toggle_reset_on_start(self):
        path = self._reset_marker_path()
        if self.reset_on_start_var.get():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("reset")
            except Exception:
                messagebox.showerror("Error", "Could not set reset-on-start marker.")
        else:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                messagebox.showerror("Error", "Could not remove reset-on-start marker.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
