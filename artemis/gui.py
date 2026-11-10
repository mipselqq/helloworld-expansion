"""Fullscreen lightweight desktop editor for the Artemis wallpaper."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import CancelledError
import json
from pathlib import Path
import re
import subprocess
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageFont, ImageDraw

from .studio import Project, Settings


def screens():
    """Read monitor positions from XRandR and native mode sizes from KDE."""
    modes={}
    try:
        data=json.loads(subprocess.check_output(["kscreen-doctor","-j"],stderr=subprocess.DEVNULL,timeout=3))
        for output in data["outputs"]:
            if output.get("enabled") and output.get("connected"):
                mode=next(m for m in output["modes"] if m["id"]==output["currentModeId"])
                width,height=mode["size"]["width"],mode["size"]["height"]
                if output.get("rotation") in (2,8):
                    width,height=height,width
                modes[output["name"]]=(width,height)
    except (OSError,subprocess.SubprocessError,ValueError,KeyError,StopIteration):
        pass
    result=[]
    try:
        listing=subprocess.check_output(["xrandr","--listmonitors"],stderr=subprocess.DEVNULL,text=True,timeout=3)
        for line in listing.splitlines()[1:]:
            found=re.search(r"(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)\s+(\S+)$",line)
            if found:
                w,h,x,y=map(int,found.group(1,2,3,4))
                result.append((x,y,w,h,*modes.get(found.group(5),(w,h))))
    except (OSError,subprocess.SubprocessError,ValueError):
        pass
    return result


def screen_for_window(monitors, bounds, fallback):
    x,y,w,h=bounds
    if not monitors:
        return fallback
    def overlap(m):
        mx,my,mw,mh,_,_=m
        return max(0,min(x+w,mx+mw)-max(x,mx))*max(0,min(y+h,my+mh)-max(y,my))
    selected=max(monitors,key=overlap)
    return selected[4],selected[5]


class StudioWindow:
    def __init__(self, root, config, fullscreen=True):
        self.root=root
        self.config=config
        self.output=config.resolve().parent/"output"
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="artemis-preview")
        self.save_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="artemis-save")
        self.project_lock=threading.Lock()
        self.project=None
        self.future=None
        self.job=None
        self.save_future=None
        self.preview_due=0
        self.revision=0
        self.displayed_revision=-1
        self.preview_cancel=None
        self.zoom=1.0
        self.center=(.5,.5)
        self.drag_origin=None
        self.monitors=screens()
        self.automatic_size=True
        self.photo=None
        self.display_image=None
        self.display_view=None
        self.placeholder_image=None
        self.preview_complete=False
        self.text_images=[]
        try:
            font_path=subprocess.check_output(["fc-match","-f","%{file}","Noto Sans"],text=True,timeout=3)
        except (OSError,subprocess.SubprocessError):
            font_path="/usr/share/fonts/TTF/DejaVuSans.ttf"
        self.font_path=font_path

        root.title("Artemis II - Wallpaper Editor")
        root.configure(bg="#080d16")
        root.geometry("1260x760")
        root.minsize(850,560)
        root.attributes("-fullscreen",fullscreen)
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.bind("<Escape>",self.toggle_fullscreen)
        root.bind("<F11>",self.toggle_fullscreen)

        style=ttk.Style(root)
        style.theme_use("clam")
        style.configure(".",background="#141c29",foreground="#e5edf7",fieldbackground="#202d3d",
                        font=("Noto Sans",11),borderwidth=0)
        style.configure("TFrame",background="#141c29")
        style.configure("TLabel",background="#141c29",foreground="#e5edf7")
        style.configure("Muted.TLabel",foreground="#a5b8ce",font=("Noto Sans",9))
        style.configure("Title.TLabel",font=("Noto Sans",19,"bold"))
        style.configure("TCheckbutton",background="#141c29",foreground="#e5edf7",padding=(2,5))
        style.map("TCheckbutton",background=[("active","#1b2838")])
        style.configure("TButton",padding=(12,9),background="#26384d")
        style.map("TButton",background=[("active","#355675"),("disabled","#202a36")])
        style.configure("Accent.TButton",background="#267eaa",foreground="white")
        style.map("Accent.TButton",background=[("active","#359dca"),("disabled","#334352")])
        style.configure("TCombobox",padding=5,arrowsize=14)
        style.configure("TEntry",padding=7)
        # Tk's Unix file picker draws its icon list using Treeview colors.
        # Defining only the global foreground makes filenames white on white.
        style.configure("Treeview",background="#202d3d",fieldbackground="#202d3d",
                        foreground="#e5edf7")
        style.map("Treeview",background=[("selected","#267eaa")],
                  foreground=[("selected","#ffffff")])
        root.option_add("*TCombobox*Listbox.background","#202d3d")
        root.option_add("*TCombobox*Listbox.foreground","#e5edf7")
        root.option_add("*TCombobox*Listbox.selectBackground","#267eaa")

        sidebar=ttk.Frame(root,padding=(24,22),width=320)
        sidebar.pack(side="left",fill="y")
        sidebar.pack_propagate(False)
        content=tk.Frame(root,bg="#080d16")
        content.pack(side="right",fill="both",expand=True)
        self.canvas=tk.Canvas(content,bg="#080d16",highlightthickness=0)
        self.canvas.pack(fill="both",expand=True,padx=15,pady=15)

        self.label(sidebar,"ARTEMIS II",size=21).pack(anchor="w")
        self.label(sidebar,"Wallpaper editor",muted=True).pack(anchor="w",pady=(2,18))
        self.variables={}
        for key,label in (("grade","Color grading"),
                          ("stars","Catalog stars in extension"),
                          ("sky_grain","Original sky texture"),
                          ("screen_grain","Video grain"),
                          ("clean_flare","Remove flare and haze"),
                          ("sun_glow","Reduce sunward sky glow")):
            var=tk.BooleanVar(value=True)
            self.variables[key]=var
            ttk.Checkbutton(sidebar,image=self.text_image(label),variable=var,command=self.request_preview).pack(anchor="w")
        self.label(sidebar,"Sharpening",muted=True).pack(anchor="w",pady=(15,4))
        self.sharp=tk.StringVar(value="Conventional")
        self.select(sidebar,self.sharp,("Conventional","Multiscale","Off"))
        self.label(sidebar,"Preview",muted=True).pack(anchor="w",pady=(15,4))
        ttk.Button(sidebar,image=self.text_image("Reset zoom"),command=self.reset_view).pack(fill="x")
        self.label(sidebar,"Wheel: zoom   Drag: pan",size=12,muted=True).pack(anchor="w",pady=(5,0))
        self.label(sidebar,"Export size",muted=True).pack(anchor="w",pady=(15,4))
        size_row=ttk.Frame(sidebar)
        size_row.pack(fill="x")
        self.width=tk.StringVar(value="3440")
        self.height=tk.StringVar(value="1440")
        for variable in (self.width,self.height):
            entry=ttk.Entry(size_row,textvariable=variable,width=8)
            entry.pack(side="left",fill="x",expand=True)
            entry.bind("<KeyRelease>",self.manual_size)
            if variable is self.width:
                self.label(size_row," x ").pack(side="left")
        ttk.Button(sidebar,image=self.text_image("Use current screen size"),command=self.detect_size).pack(fill="x",pady=(7,0))
        self.save_button=ttk.Button(sidebar,image=self.text_image("Save wallpaper"),style="Accent.TButton",command=self.save)
        self.save_button.pack(fill="x",pady=(18,0))
        ttk.Button(sidebar,image=self.text_image("Close"),command=self.close).pack(fill="x",pady=(8,0))
        self.label(sidebar,"Esc / F11: toggle fullscreen",muted=True).pack(anchor="w",pady=(16,4))
        self.status=ttk.Label(sidebar)
        self.set_status("Preparing image...")
        self.status.pack(anchor="w",pady=(6,0))
        self.save_status=ttk.Label(sidebar)
        self.save_status.pack(anchor="w",pady=(6,0))
        self.canvas.bind("<Configure>",self.canvas_changed)
        self.canvas.bind("<MouseWheel>",self.wheel)
        self.canvas.bind("<Button-4>",self.wheel)
        self.canvas.bind("<Button-5>",self.wheel)
        self.canvas.bind("<ButtonPress-1>",self.drag_start)
        self.canvas.bind("<B1-Motion>",self.drag_move)
        self.canvas.bind("<ButtonRelease-1>",self.drag_end)
        root.bind("<Configure>",self.window_changed)
        root.after(300,self.detect_size)
        self.show_placeholder()
        self.request_preview()
        root.after(80,self.poll)

    def text_image(self, text, size=14, muted=False, keep=True):
        """Render readable labels even on Tk builds with only bitmap fonts."""
        try:
            font=ImageFont.truetype(self.font_path,size)
        except OSError:
            font=ImageFont.load_default()
        lines=[]
        for line in str(text).splitlines() or [""]:
            while len(line)>35:
                lines.append(line[:35])
                line=line[35:]
            lines.append(line)
        text="\n".join(lines)
        bounds=ImageDraw.Draw(Image.new("RGB",(1,1))).multiline_textbbox((0,0),text,font=font,spacing=4)
        img=Image.new("RGBA",(max(1,bounds[2]-bounds[0]+2),max(1,bounds[3]-bounds[1]+2)))
        ImageDraw.Draw(img).multiline_text((1-bounds[0],1-bounds[1]),text,font=font,
                                         fill="#a5b8ce" if muted else "#e5edf7",spacing=4)
        result=ImageTk.PhotoImage(img,master=self.root)
        if keep:
            self.text_images.append(result)
        return result

    def label(self, parent, text, size=14, muted=False):
        return ttk.Label(parent,image=self.text_image(text,size,muted))

    def set_status(self, text):
        self.status_image=self.text_image(text,size=13,muted=True,keep=False)
        self.status.configure(image=self.status_image)

    def set_save_status(self, text):
        self.save_status_image=self.text_image(text,size=13,muted=True,keep=False)
        self.save_status.configure(image=self.save_status_image)

    def select(self, parent, variable, options):
        button=ttk.Menubutton(parent,direction="below")
        menu=tk.Menu(button,tearoff=False,bg="#202d3d",activebackground="#355675",borderwidth=0)
        button.configure(menu=menu)
        def choose(value):
            variable.set(value)
            button.configure(image=self.text_image(value+"   v"))
            self.request_preview()
        button.configure(image=self.text_image(variable.get()+"   v"))
        for value in options:
            menu.add_command(image=self.text_image(value),command=lambda value=value:choose(value))
        button.pack(fill="x")

    def show_placeholder(self):
        path=self.output/"preview.jpg"
        if not path.exists():
            path=self.config.resolve().parent/"reference"/"preview.jpg"
        if path.exists():
            with Image.open(path) as image:
                self.placeholder_image=image.copy()
            self.update_placeholder()

    def update_placeholder(self):
        if self.preview_complete or self.placeholder_image is None:
            return
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        if cw<10 or ch<10:
            return
        try:
            size=self.resolution()
        except ValueError:
            return
        w,h=size
        scale=min(cw/w,ch/h)
        target=(max(1,round(w*scale)),max(1,round(h*scale)))
        thumb=self.placeholder_image.resize(target,Image.Resampling.LANCZOS)
        canvas=Image.new("RGB",(cw,ch),(8,13,22))
        canvas.paste(thumb,((cw-target[0])//2,(ch-target[1])//2))
        self.display_view=((.5,.5),1.0,size)
        self.show(canvas)
        if self.zoom!=1.0 or self.center!=(.5,.5):
            self.interactive_view()

    def show(self, image):
        self.display_image=image
        self.refresh_canvas()

    def refresh_canvas(self):
        if self.display_image is None:
            return
        width=max(1,self.canvas.winfo_width())
        height=max(1,self.canvas.winfo_height())
        image=self.display_image.copy()
        image.thumbnail((width,height),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(image,master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,
                                 image=self.photo,anchor="center")

    def interactive_view(self):
        """Show an immediate temporary zoom/pan while the detailed render runs."""
        if self.display_view is None or self.display_image is None:
            return
        old_center,old_zoom,size=self.display_view
        try:
            if size!=self.resolution():
                return
        except ValueError:
            return
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        if self.display_image.size!=(cw,ch):
            return
        w,h=size
        old_scale=min(cw/w,ch/h)*old_zoom
        ratio=old_zoom/self.zoom
        tx=cw/2+(self.center[0]-old_center[0])*w*old_scale-ratio*cw/2
        ty=ch/2+(self.center[1]-old_center[1])*h*old_scale-ratio*ch/2
        image=self.display_image.transform((cw,ch),Image.Transform.AFFINE,
                  (ratio,0,tx,0,ratio,ty),resample=Image.Resampling.BILINEAR,
                  fillcolor=(8,13,22))
        self.photo=ImageTk.PhotoImage(image,master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(cw/2,ch/2,image=self.photo,anchor="center")

    def toggle_fullscreen(self, event=None):
        self.root.attributes("-fullscreen",not self.root.attributes("-fullscreen"))

    def window_changed(self, event):
        if event.widget is self.root and self.automatic_size:
            if hasattr(self,"monitor_timer"):
                self.root.after_cancel(self.monitor_timer)
            self.monitor_timer=self.root.after(600,self.detect_size)

    def canvas_changed(self, event):
        if hasattr(self,"canvas_size") and self.canvas_size==(event.width,event.height):
            return
        self.canvas_size=(event.width,event.height)
        if self.preview_complete:
            self.refresh_canvas()
        else:
            self.update_placeholder()
        self.request_preview()

    def detect_size(self):
        self.root.update_idletasks()
        bounds=(self.root.winfo_rootx(),self.root.winfo_rooty(),self.root.winfo_width(),self.root.winfo_height())
        size=screen_for_window(self.monitors,bounds,(self.root.winfo_screenwidth(),self.root.winfo_screenheight()))
        if (self.width.get(),self.height.get())!=(str(size[0]),str(size[1])):
            self.width.set(str(size[0]))
            self.height.set(str(size[1]))
            self.request_preview()
        self.automatic_size=True

    def manual_size(self, event=None):
        self.automatic_size=False
        self.request_preview()

    def view_scale(self):
        try:
            w,h=self.resolution()
        except ValueError:
            return None
        return min(max(1,self.canvas.winfo_width())/w,max(1,self.canvas.winfo_height())/h)*self.zoom

    def wheel(self,event):
        scale=self.view_scale()
        if scale is None:
            return
        steps=(1 if event.num==4 else -1) if getattr(event,"num",None) in (4,5) else event.delta/120
        if not steps:
            return
        new_zoom=max(1.0,min(16.0,self.zoom*1.25**steps))
        if new_zoom==self.zoom:
            return
        w,h=self.resolution()
        offset=(event.x-self.canvas.winfo_width()/2,event.y-self.canvas.winfo_height()/2)
        new_scale=scale*new_zoom/self.zoom
        self.center=tuple(max(0,min(1,c+d/n*(1/scale-1/new_scale)))
                          for c,d,n in zip(self.center,offset,(w,h)))
        self.zoom=new_zoom
        if self.zoom==1:
            self.center=(.5,.5)
        self.interactive_view()
        self.request_preview()

    def drag_start(self,event):
        self.drag_origin=(event.x,event.y)
        self.canvas.configure(cursor="fleur")

    def drag_move(self,event):
        if self.drag_origin is None or self.zoom<=1:
            return
        scale=self.view_scale()
        if scale is None:
            return
        w,h=self.resolution()
        dx,dy=event.x-self.drag_origin[0],event.y-self.drag_origin[1]
        self.center=(max(0,min(1,self.center[0]-dx/(scale*w))),
                     max(0,min(1,self.center[1]-dy/(scale*h))))
        self.drag_origin=(event.x,event.y)
        self.interactive_view()
        self.request_preview()

    def drag_end(self,event):
        self.drag_origin=None
        self.canvas.configure(cursor="")

    def reset_view(self):
        self.zoom=1.0
        self.center=(.5,.5)
        self.interactive_view()
        self.request_preview()

    def resolution(self):
        try:
            width,height=int(self.width.get()),int(self.height.get())
        except ValueError:
            raise ValueError("Enter width and height in pixels.") from None
        if width<1 or height<1 or width>16000 or height>16000 or width*height>100_000_000:
            raise ValueError("Dimensions must be 1 to 16000 pixels and at most 100 MP.")
        return width,height

    def settings(self):
        sharp={"Conventional":"conventional","Multiscale":"multiscale","Off":"off"}[self.sharp.get()]
        return Settings(**{k:v.get() for k,v in self.variables.items()},sharpness=sharp)

    def request_preview(self, event=None):
        self.revision+=1
        self.preview_due=time.monotonic()+.25
        if self.preview_cancel is not None:
            self.preview_cancel.set()

    def ensure_project(self):
        if self.project is None:
            with self.project_lock:
                if self.project is None:
                    self.project=Project(self.config)
        return self.project

    def render_preview(self, options, size, dimensions, center, zoom, cancel):
        return self.ensure_project().preview_view(options,size,dimensions,center,zoom,cancel)

    def render_save(self, options, size, path):
        self.ensure_project().save(options,size,path)
        return path

    def save(self):
        try:
            size=self.resolution()
        except ValueError as error:
            messagebox.showerror("Image size",str(error),parent=self.root)
            return
        self.output.mkdir(parents=True,exist_ok=True)
        target=filedialog.asksaveasfilename(parent=self.root,title="Save wallpaper",initialdir=self.output,
                  initialfile=f"artemis_ii_{size[0]}x{size[1]}.png",defaultextension=".png",
                  filetypes=(("PNG, 16-bit","*.png"),("TIFF, 16-bit","*.tif *.tiff")))
        if not target:
            return
        options=self.settings()
        self.save_button.state(["disabled"])
        self.set_save_status("Rendering export with current settings...")
        self.save_future=self.save_executor.submit(self.render_save,options,size,Path(target))

    def poll(self):
        if self.save_future is not None and self.save_future.done():
            try:
                path=self.save_future.result()
                self.set_save_status(f"Saved: {path.name}")
            except Exception as error:
                self.set_save_status("Save failed")
                messagebox.showerror("Save failed",str(error),parent=self.root)
            self.save_button.state(["!disabled"])
            self.save_future=None
        if self.future is not None and self.future.done():
            revision,center,zoom,size=self.job
            try:
                value=self.future.result()
                if revision==self.revision:
                    self.display_view=(center,zoom,size)
                    self.preview_complete=True
                    self.show(value)
                    self.displayed_revision=revision
                    self.set_status(f"Preview ready  |  zoom {self.zoom:.1f}x")
            except CancelledError:
                pass
            except Exception as error:
                if revision==self.revision:
                    self.set_status(str(error))
                    self.displayed_revision=revision
            self.future=None
            self.job=None
            self.preview_cancel=None
        if self.future is None:
            if self.displayed_revision!=self.revision and time.monotonic()>=self.preview_due:
                try:
                    size=self.resolution()
                except ValueError as error:
                    self.set_status(str(error))
                    self.displayed_revision=self.revision
                else:
                    self.set_status("Updating preview...")
                    self.job=(self.revision,self.center,self.zoom,size)
                    dimensions=(max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height()))
                    self.preview_cancel=threading.Event()
                    self.future=self.executor.submit(self.render_preview,self.settings(),size,dimensions,
                                                     self.center,self.zoom,self.preview_cancel)
        self.root.after(80,self.poll)

    def close(self):
        if self.preview_cancel is not None:
            self.preview_cancel.set()
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.save_executor.shutdown(wait=False,cancel_futures=True)
        self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config.toml"))
    parser.add_argument("--windowed",action="store_true")
    args=parser.parse_args()
    root=tk.Tk()
    StudioWindow(root,args.config,fullscreen=not args.windowed)
    root.mainloop()


if __name__=="__main__":
    main()
