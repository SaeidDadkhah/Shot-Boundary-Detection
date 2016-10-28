import tkinter.filedialog as filedialog
from tkinter import *
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import _thread
import time
import csv
from ShotBoundaryDetection import ShotBoundaryDetection


class SBDGUI:
    def __init__(self):
        self.__sbd = ShotBoundaryDetection()
        self.__playing_backward = False
        self.__playing_forward = False

        # Making root window
        self.root = Tk()
        self.root.title("Shot Boundary Detection")

        # Controlling size of root window
        # self.root.resizable(0, 0)
        width = 932
        height = 607
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry("{width}x{height}+{x}+{y}".format(width=width,
                                                             height=height,
                                                             x=int((screen_width - width) / 2),
                                                             y=int((screen_height - height - 50) / 2)))

        # Making menu
        menu_main = Menu(self.root)

        menu_file = Menu(menu_main, tearoff=0)
        menu_file.add_command(label="Open", command=lambda: self.open_file())
        menu_file.add_separator()
        menu_file.add_command(label="Close", command=exit)
        menu_main.add_cascade(label="File", menu=menu_file)

        self.root.config(menu=menu_main)

        # Making video frame
        # Some of values of other frames are set here.
        frame_shot_boundaries_width = 280
        frame_control_height = 100
        frame_video = LabelFrame(self.root, text="Video")
        frame_video.place(width=-frame_shot_boundaries_width,
                          height=-frame_control_height,
                          x=0,
                          y=0,
                          relheight=1,
                          relwidth=1,
                          relx=0,
                          rely=0)
        self.label_video = Label(frame_video)
        self.label_video.place(relwidth=1,
                               relheight=1,
                               relx=0,
                               rely=0)

        frame_shot_boundaries = LabelFrame(self.root, text="Shot Boundaries")
        frame_shot_boundaries.place(width=frame_shot_boundaries_width,
                                    height=0,
                                    x=-frame_shot_boundaries_width,
                                    y=0,
                                    relwidth=0,
                                    relheight=1,
                                    relx=1,
                                    rely=0)

        scrollbar_shot_boundaries = Scrollbar(frame_shot_boundaries)
        scrollbar_shot_boundaries.place(relwidth=1,
                                        relheight=1,
                                        relx=0,
                                        rely=0)

        self.listbox_shot_boundaries = Listbox(frame_shot_boundaries,
                                               yscrollcommand=scrollbar_shot_boundaries.set)
        self.listbox_shot_boundaries.place(width=-18,
                                           relwidth=1,
                                           relheight=1)
        self.listbox_shot_boundaries.bind('<<ListboxSelect>>', lambda event: self.__shot_selected(event))
        scrollbar_shot_boundaries.config(command=self.listbox_shot_boundaries.yview)

        frame_control = LabelFrame(self.root, text="Toolbox")
        frame_control.place(width=-frame_shot_boundaries_width,
                            height=frame_control_height,
                            x=0,
                            y=-frame_control_height,
                            relheight=0,
                            relwidth=1,
                            relx=0,
                            rely=1)

        self.__button_previous_frame = Button(frame_control,
                                              text="Previous Frame",
                                              state=DISABLED,
                                              command=lambda: (self.__pause(), self.__previous_frame()))
        self.__button_previous_frame.place(x=-200,
                                           relx=0.5)

        self.__button_play_backward = Button(frame_control,
                                             text="Play Backward",
                                             state=DISABLED,
                                             command=lambda: self.__play_backward())
        self.__button_play_backward.place(x=-105,
                                          relx=0.5)

        self.__button_play_forward = Button(frame_control,
                                            text="Play",
                                            state=DISABLED,
                                            command=lambda: self.__play_forward())
        self.__button_play_forward.place(x=-15,
                                         relx=0.5)

        self.__button_next_frame = Button(frame_control,
                                          text="Next Frame",
                                          state=DISABLED,
                                          command=lambda: (self.__pause(), self.__next_frame()))
        self.__button_next_frame.place(x=20,
                                       relx=0.5)

        self.__button_process_all = Button(frame_control,
                                           text="Detect Boundaries",
                                           state=DISABLED,
                                           command=lambda: self.__detect())
        self.__button_process_all.place(x=100,
                                        relx=0.5)

        self.__button_export = Button(frame_control,
                                      text="Export",
                                      state=DISABLED,
                                      command=lambda: self.__export())
        self.__button_export.place(x=210,
                                   relx=0.5)

        progress_bar_height = 31
        self.progress_bar = ttk.Progressbar(frame_control,
                                            orient=HORIZONTAL,
                                            length=700)
        self.progress_bar.place(width=-6,
                                x=3,
                                y=-progress_bar_height - 8,
                                height=progress_bar_height,
                                relwidth=1,
                                relx=0,
                                rely=1)

        self.label_status = Label(frame_control, text="Video is not available!")
        self.label_status.place(x=0,
                                y=-progress_bar_height - 3,
                                relx=0.2,
                                rely=1)

        # self.open_test()
        self.root.mainloop()

    def __control_buttons(self,
                          previous_frame=None,
                          play_backward=None,
                          play_forward=None,
                          next_frame=None,
                          process_all=None,
                          export=None):
        if previous_frame is not None:
            self.__button_previous_frame.config(state=previous_frame)
        if play_backward is not None:
            self.__button_play_backward.config(state=play_backward)
        if play_forward is not None:
            self.__button_play_forward.config(state=play_forward)
        if next_frame is not None:
            self.__button_next_frame.config(state=next_frame)
        if next_frame is not None:
            self.__button_process_all.config(state=process_all)
        if export is not None:
            self.__button_export.config(state=export)

    def open_file(self):
        input_name = filedialog.askopenfile()
        self.__control_buttons(previous_frame=NORMAL,
                               play_backward=NORMAL,
                               play_forward=NORMAL,
                               next_frame=NORMAL,
                               process_all=NORMAL)
        if input_name is not None:
            self.__sbd.open_video(input_name.name)
            self.__set_progress(0, "Video is not processed!")

    def __previous_frame(self):
        if self.__sbd.video_is_available():
            frame = self.__sbd.previous_frame(rearrange=True)
            if frame is None:
                return False
            image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label_video.configure(image=image)
            self.label_video.image = image
            return True
        else:
            print("Video is not available!")
            return False

    def __play_backward(self):
        if self.__playing_forward:
            return
        self.__control_buttons(play_forward=DISABLED,
                               process_all=DISABLED)
        self.__playing_backward = not self.__playing_backward
        available_frame = True
        while self.__playing_backward and available_frame:
            available_frame = self.__previous_frame()
            self.root.update()
        self.__playing_backward = False
        self.__control_buttons(play_forward=NORMAL,
                               process_all=NORMAL)

    def __pause(self):
        self.__playing_backward = False
        self.__playing_forward = False
        self.__control_buttons(play_forward=NORMAL,
                               play_backward=NORMAL)

    def __play_forward(self):
        if self.__playing_backward:
            return
        self.__control_buttons(play_backward=DISABLED,
                               process_all=DISABLED)
        self.__playing_forward = not self.__playing_forward
        available_frame = True
        while self.__playing_forward and available_frame:
            available_frame = self.__next_frame()
            self.root.update()
            time.sleep(0.01)
        self.__playing_forward = False
        self.__control_buttons(play_backward=NORMAL,
                               process_all=NORMAL)

    def __next_frame(self):
        if self.__sbd.video_is_available():
            frame = self.__sbd.next_frame(rearrange=True)
            if frame is None:
                return False
            image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label_video.configure(image=image)
            self.label_video.image = image
            return True
        else:
            print("Video is not available!")
            return False

    def __detect(self):
        if self.__sbd.video_is_available():
            self.__control_buttons(previous_frame=DISABLED,
                                   play_backward=DISABLED,
                                   play_forward=DISABLED,
                                   next_frame=DISABLED,
                                   process_all=DISABLED,
                                   export=DISABLED)
            _thread.start_new_thread(self.__sbd.detect,
                                     (lambda progress, status: self.__set_progress(progress, status),
                                      lambda: self.__finish()))
            # self.__sbd.detect(lambda progress, status: self.__set_progress(progress, status),
            #  lambda: self.__finish())
        else:
            print("Video is not available!")

    def __set_progress(self, progress, status):
        self.progress_bar['value'] = progress
        self.label_status['text'] = status

    def __finish(self):
        self.listbox_shot_boundaries.delete(0, END)
        for index, shot in enumerate(self.__sbd.sb):
            if shot['transition'] == 'cut':
                self.listbox_shot_boundaries. \
                    insert(index,
                           "Cut: {h}:{m}:{s}.{ms} ({f})".format(h=shot['cut_time'].hour,
                                                                m=shot['cut_time'].minute,
                                                                s=shot['cut_time'].second,
                                                                ms=shot['cut_time'].microsecond,
                                                                f=shot['cut_frame']))
            elif shot['transition'] == 'gradual':
                self.listbox_shot_boundaries. \
                    insert(index,
                           "Gradual: {sh}:{sm}:{ss}.{sms} ({sf})"
                           " - {eh}:{em}:{es}.{ems} ({ef})".format(sh=shot['start_time'].hour,
                                                                   sm=shot['start_time'].minute,
                                                                   ss=shot['start_time'].second,
                                                                   sms=shot['start_time'].microsecond,
                                                                   sf=shot['start_frame'],
                                                                   eh=shot['end_time'].hour,
                                                                   em=shot['end_time'].minute,
                                                                   es=shot['end_time'].second,
                                                                   ems=shot['end_time'].microsecond,
                                                                   ef=shot['end_frame'], ))
        self.__set_progress(100, "Video is processed.")
        self.__control_buttons(previous_frame=NORMAL,
                               play_backward=NORMAL,
                               play_forward=NORMAL,
                               next_frame=NORMAL,
                               process_all=NORMAL,
                               export=NORMAL)
        # import datetime
        # print("{s}: {t}".format(s="finish",
        #                         t=datetime.datetime.now()))

    # noinspection PyUnusedLocal
    def __shot_selected(self, event):
        if len(self.listbox_shot_boundaries.curselection()) > 0:
            index = int(self.listbox_shot_boundaries.curselection()[0])
            print(index)
            if self.__sbd.sb[index]['transition'] == 'cut':
                self.__sbd.set_frame(self.__sbd.sb[index]['cut_frame'])
            elif self.__sbd.sb[index]['transition'] == 'gradual':
                self.__sbd.set_frame(self.__sbd.sb[index]['start_frame'])
            self.__next_frame()

    def __export(self):
        with open('test.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Transition',
                                                   'Start Frame',
                                                   'Start Time',
                                                   'End Frame',
                                                   'End Time'])
            writer.writeheader()
            for shot in self.__sbd.sb:
                if shot['transition'] == 'cut':
                    writer.writerow({'Transition': 'Cut',
                                     'Start Frame': shot['cut_frame'],
                                     'Start Time': "{h}:{m}:{s}.{ms}".format(h=shot['cut_time'].hour,
                                                                             m=shot['cut_time'].minute,
                                                                             s=shot['cut_time'].second,
                                                                             ms=shot['cut_time'].microsecond),
                                     'End Frame': '---',
                                     'End Time': '---'})
                elif shot['transition'] == 'gradual':
                    writer.writerow({'Transition': 'Gradual',
                                     'Start Frame': shot['start_frame'],
                                     'Start Time': "{h}:{m}:{s}.{ms}".format(h=shot['start_time'].hour,
                                                                             m=shot['start_time'].minute,
                                                                             s=shot['start_time'].second,
                                                                             ms=shot['start_time'].microsecond),
                                     'End Frame': shot['end_frame'],
                                     'End Time': "{h}:{m}:{s}.{ms}".format(h=shot['end_time'].hour,
                                                                           m=shot['end_time'].minute,
                                                                           s=shot['end_time'].second,
                                                                           ms=shot['end_time'].microsecond)})

    def open_test(self):
        self.__sbd.open_video("test.mov")
        self.label_status['text'] = "Video is not processed!"
        self.__button_process_all.config(state=NORMAL)
        self.__next_frame()


if __name__ == "__main__":
    sbdgui = SBDGUI()
