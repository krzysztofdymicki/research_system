import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys
import shutil

try:
    from src.orchestrator import SearchOptions, run_search_and_save
    from src.db import init_db, get_recent_results, reset_db
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.orchestrator import SearchOptions, run_search_and_save
    from src.db import init_db, get_recent_results, reset_db


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

        # Results tab
        res = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(res, text="Results")

        ttk.Label(res, text="Filter (query substring):").grid(row=0, column=0, sticky=tk.W)
        self.filter_var = tk.StringVar()
        ttk.Entry(res, textvariable=self.filter_var, width=30).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(res, text="Refresh", command=self.refresh_results).grid(row=0, column=2, padx=5)

        self.tree = ttk.Treeview(
            res,
            columns=("title", "source", "query", "relevance", "label", "has_md"),
            show="headings",
            height=12,
        )
        self.tree.heading("title", text="Title")
        self.tree.heading("source", text="Source")
        self.tree.heading("query", text="Query")
        self.tree.heading("relevance", text="Rel.")
        self.tree.heading("label", text="Label")
        self.tree.heading("has_md", text="Markdown")
        self.tree.column("title", width=360)
        self.tree.column("source", width=80)
        self.tree.column("query", width=180)
        self.tree.column("relevance", width=60, anchor=tk.E)
        self.tree.column("label", width=80)
        self.tree.column("has_md", width=90)
        self.tree.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 8))
        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)

        # Analysis controls (operate on all results; filter is just for browsing)
        ctrl = ttk.Frame(res)
        ctrl.grid(row=2, column=0, columnspan=3, sticky="ew")
        ttk.Label(ctrl, text="AI threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=70)
        ttk.Spinbox(ctrl, from_=0, to=100, textvariable=self.threshold_var, width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(ctrl, text="Run AI Analysis", command=self.on_run_ai).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Download PDFs (kept)", command=self.on_download_kept).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(ctrl, text="Extract MD (kept)", command=self.on_extract_kept).pack(side=tk.LEFT, padx=(6, 0))
        self.only_kept_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Show only kept", variable=self.only_kept_var, command=self.refresh_results).pack(side=tk.LEFT, padx=10)

        self.detail = tk.Text(res, height=8, wrap=tk.WORD)
        self.detail.grid(row=3, column=0, columnspan=3, sticky="nsew")

        res.rowconfigure(1, weight=1)
        res.columnconfigure(0, weight=1)
        res.columnconfigure(1, weight=0)
        res.columnconfigure(2, weight=0)

        self.refresh_results()

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
        self.status_var.set("Running ...")

        def work():
            try:
                total, saved = run_search_and_save(opts)
                self.status_var.set(f"Done. Fetched {total} items. Saved {saved}.")
            except Exception as e:
                self.status_var.set("Error. See console.")
                messagebox.showerror("Error", str(e))
            finally:
                self.run_btn.config(state=tk.NORMAL)

        threading.Thread(target=work, daemon=True).start()

    def refresh_results(self):
        conn = init_db()
        rows = get_recent_results(
            conn,
            query_filter=self.filter_var.get() or None,
            only_kept=self.only_kept_var.get(),
            limit=200,
        )
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in rows:
            rel = r.get("relevance_score")
            label = r.get("relevance_label") or ""
            rel_disp = "" if rel is None else f"{int(rel)}"
            self.tree.insert(
                "",
                tk.END,
                iid=str(r["search_result_id"]),
                values=(r["title"], r["source"], r["query"], rel_disp, label, "yes" if r["has_markdown"] else "no"),
            )
        self._rows_by_id = {str(r["search_result_id"]): r for r in rows}

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
            f"Query: {r['query']}",
            f"URL: {r['url']}",
            f"PDF: {r.get('pdf_path') or r.get('pdf_url')}",
            f"Relevance: {r.get('relevance_score')}",
            f"Label: {r.get('relevance_label')}",
            f"Authors: {authors}",
            f"Abstract: {r.get('abstract') or ''}",
        ]
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", "\n".join(lines))

    def on_run_ai(self):
        threshold = self.threshold_var.get()
        def work():
            try:
                from src.orchestrator import analyze_and_filter
                analyzed, kept = analyze_and_filter(None, threshold)
                self.status_var.set(f"AI analyzed {analyzed}, kept {kept}.")
            except Exception as e:
                messagebox.showerror("AI Error", str(e))
            finally:
                self.refresh_results()

        threading.Thread(target=work, daemon=True).start()

    def on_download_kept(self):
        def work():
            try:
                from src.orchestrator import download_pdfs_for_kept
                attempted, downloaded = download_pdfs_for_kept(None)
                self.status_var.set(f"PDFs: attempted {attempted}, downloaded {downloaded}.")
            except Exception as e:
                messagebox.showerror("Download Error", str(e))
            finally:
                self.refresh_results()
        threading.Thread(target=work, daemon=True).start()

    def on_extract_kept(self):
        def work():
            try:
                from src.orchestrator import extract_markdown_for_kept
                attempted, extracted = extract_markdown_for_kept(None)
                self.status_var.set(f"Markdown: attempted {attempted}, extracted {extracted}.")
            except Exception as e:
                messagebox.showerror("Extract Error", str(e))
            finally:
                self.refresh_results()
        threading.Thread(target=work, daemon=True).start()

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
