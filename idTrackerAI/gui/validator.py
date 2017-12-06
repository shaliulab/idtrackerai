from __future__ import absolute_import, division, print_function
import kivy

from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from visualise_video import VisualiseVideo
from kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('../')
sys.path.append('../utils')
import numpy as np
import cv2

from video import Video
from py_utils import getExistentFiles, get_spaced_colors_util
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from generate_groundtruth import generate_groundtruth
from compute_groundtruth_statistics import get_accuracy_wrt_groundtruth

class Validator(BoxLayout):
    def __init__(self, chosen_video = None,
                deactivate_validation = None,
                **kwargs):
        super(Validator, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_VALIDATION
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_VALIDATION = deactivate_validation

        self.visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
        self.warning_popup = Popup(title = 'Warning',
                            content = CustomLabel(text = 'The video has not been tracked yet. Track it before performing validation.'),
                            size_hint = (.3,.3))
        self.loading_popup = Popup(title='Loading',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)


    def show_saving(self, *args):
        self.popup_saving = Popup(title='Saving',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))
        self.popup_saving.open()

    def create_count_bad_crossing_popup(self):
        self.wc_popup_container = BoxLayout()
        self.wc_identity_box = BoxLayout(orientation="vertical")
        self.wc_label = CustomLabel(text='Type the identity associated to a badly corrected crossing')
        self.wc_identity_box.add_widget(self.wc_label)
        self.wc_identity_input = TextInput(text ='', multiline=False)
        self.wc_identity_box.add_widget(self.wc_identity_input)
        self.wc_popup_container.add_widget(self.wc_identity_box)
        self.wc_popup = Popup(title = 'Count wrong crossings',
                            content = self.wc_popup_container,
                            size_hint = (.4, .4))

    def on_enter_wrong_crossing_identity(self, value):
        self.wrong_crossing_counter[int(self.wc_identity_input.text)] += 1
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.wc_popup.dismiss()

    def create_choose_list_of_blobs_popup(self):
        self.lob_container = BoxLayout()
        self.lob_box = BoxLayout(orientation="vertical")
        self.lob_label = CustomLabel(text='We detected two different trajectory files. Which one do you want to use for validation?')
        self.lob_btns_container = BoxLayout()
        self.lob_btn1 = Button(text = "With gaps")
        self.lob_btn2 = Button(text = "Without gaps")
        self.lob_btns_container.add_widget(self.lob_btn1)
        self.lob_btns_container.add_widget(self.lob_btn2)
        self.lob_box.add_widget(self.lob_label)
        self.lob_box.add_widget(self.lob_btns_container)
        self.lob_container.add_widget(self.lob_box)
        self.choose_list_of_blobs_popup = Popup(title = 'Choose validation trajectories',
                            content = self.lob_container,
                            size_hint = (.4, .4))

    def show_loading_text(self, *args):
        self.lob_label.text = "Loading..."

    def on_choose_list_of_blobs_btns_press(self, instance):
        if instance.text == 'With gaps':
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_path
        else:
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_no_gaps_path)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_no_gaps_path
        if not self.list_of_blobs.blobs_are_connected:
            self.list_of_blobs.reconnect()
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.choose_list_of_blobs_popup.dismiss()
        self.populate_validation_tab()

    def populate_validation_tab(self):
        self.blobs_in_video = self.list_of_blobs.blobs_in_video
        self.count_scrollup = 0
        self.scale = 1
        self.wrong_crossing_counter = {identity: 0 for identity in range(1, CHOSEN_VIDEO.video.number_of_animals + 1)}
        self.create_count_bad_crossing_popup()
        self.wc_identity_input.bind(on_text_validate = self.on_enter_wrong_crossing_identity)
        self.loading_popup.dismiss()
        self.init_segmentZero()

    def get_first_frame(self):
        if not hasattr(CHOSEN_VIDEO.video, 'first_frame_first_global_fragment'):
            CHOSEN_VIDEO.video._first_frame_first_global_fragment = CHOSEN_VIDEO.old_video.first_frame_first_global_fragment
        return CHOSEN_VIDEO.video.first_frame_first_global_fragment

    def do(self, *args):
        try:
            if CHOSEN_VIDEO.processes_to_restore is not None and CHOSEN_VIDEO.processes_to_restore['assignment']:
                CHOSEN_VIDEO.video.__dict__.update(CHOSEN_VIDEO.old_video.__dict__)
            if  CHOSEN_VIDEO.processes_to_restore is not None\
                and 'crossings' in CHOSEN_VIDEO.processes_to_restore\
                and CHOSEN_VIDEO.processes_to_restore['crossings']:
                self.create_choose_list_of_blobs_popup()
                self.lob_btn1.bind(on_press = self.show_loading_text)
                self.lob_btn2.bind(on_press = self.show_loading_text)
                self.lob_btn1.bind(on_release = self.on_choose_list_of_blobs_btns_press)
                self.lob_btn2.bind(on_release = self.on_choose_list_of_blobs_btns_press)
                self.choose_list_of_blobs_popup.open()
            else:
                self.loading_popup.open()
                self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path)
                self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_path
                if not self.list_of_blobs.blobs_are_connected:
                    self.list_of_blobs.reconnect()
                self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
                self._keyboard.bind(on_key_down=self._on_keyboard_down)
                self.populate_validation_tab()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            self.warning_popup.open()

    def init_segmentZero(self):
        self.add_widget(self.visualiser)
        self.colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals)
        print(self.colors)
        self.button_box = BoxLayout(orientation='vertical', size_hint=(.3,1.))
        self.add_widget(self.button_box)
        self.next_cross_button = Button(id='crossing_btn', text='Go to next crossing', size_hint=(1,1))
        self.next_cross_button.bind(on_press=self.go_to_next_crossing)
        self.button_box.add_widget(self.next_cross_button)
        self.previous_cross_button = Button(id='crossing_btn', text='Go to previous crossing', size_hint=(1,1))
        self.previous_cross_button.bind(on_press=self.go_to_previous_crossing)
        self.button_box.add_widget(self.previous_cross_button)
        self.go_to_first_global_fragment_button = Button(id='back_to_first_gf_btn', text='First global fragment', size_hint=(1,1))
        self.go_to_first_global_fragment_button.bind(on_press = self.go_to_first_global_fragment)
        self.button_box.add_widget(self.go_to_first_global_fragment_button)
        self.save_groundtruth_btn = Button(id='save_groundtruth_btn', text='Save updated identities',size_hint = (1,1))
        self.save_groundtruth_btn.bind(on_press=self.show_saving)
        self.save_groundtruth_btn.bind(on_release=self.save_groundtruth_list_of_blobs)
        self.save_groundtruth_btn.disabled = True
        self.button_box.add_widget(self.save_groundtruth_btn)
        self.compute_accuracy_button = Button(id = "compute_accuracy_button", text = "Compute accuracy", size_hint  = (1.,1.))
        self.compute_accuracy_button.disabled = False
        self.compute_accuracy_button.bind(on_press = self.compute_and_save_session_accuracy_wrt_groundtruth_APP)
        self.button_box.add_widget(self.compute_accuracy_button)
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.writeIds, frame_index_to_start = self.get_first_frame())

    def go_to_next_crossing(self,instance):
        non_crossing = True
        frame_index = int(self.visualiser.video_slider.value)

        while non_crossing == True:
            if frame_index < CHOSEN_VIDEO.video.number_of_frames - 1:
                frame_index = frame_index + 1
                blobs_in_frame = self.blobs_in_video[frame_index]
                for blob in blobs_in_frame:
                    if not blob.is_an_individual:
                        non_crossing = False
                        self.visualiser.video_slider.value = frame_index
                        self.visualiser.visualise(frame_index, func = self.writeIds)
            else:
                break

    def go_to_previous_crossing(self,instance):
        non_crossing = True
        frame_index = int(self.visualiser.video_slider.value)

        while non_crossing == True:
            if frame_index > 0:
                frame_index = frame_index - 1
                blobs_in_frame = self.blobs_in_video[frame_index]
                for blob in blobs_in_frame:
                    if not blob.is_an_individual:
                        non_crossing = False
                        self.visualiser.video_slider.value = frame_index
                        self.visualiser.visualise(frame_index, func = self.writeIds)
            else:
                break

    def go_to_first_global_fragment(self, instance):
        self.visualiser.visualise(self.get_first_frame(), func = self.writeIds)
        self.visualiser.video_slider.value = self.get_first_frame()

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):

        frame_index = int(self.visualiser.video_slider.value)
        if keycode[1] == 'left':
            frame_index -= 1
        elif keycode[1] == 'right':
            frame_index += 1
        elif keycode[1] == 'c':
            self.wc_popup.open()
        self.visualiser.video_slider.value = frame_index
        self.visualiser.visualise(frame_index, func = self.writeIds)
        return True

    @staticmethod
    def get_clicked_blob(point, contours):
        """
        Get the contour that contains point
        """
        indices = [i for i, cnt in enumerate(contours) if cv2.pointPolygonTest(cnt, tuple(point), measureDist = False) >= 0]
        if len(indices) != 0:
            return indices[0]
        else:
            return None

    def apply_affine_transform_on_point(self, affine_transform_matrix, point):
        R = affine_transform_matrix[:,:-1]
        T = affine_transform_matrix[:,-1]
        return np.dot(R, np.squeeze(point)) + T

    def apply_inverse_affine_transform_on_point(self, affine_transform_matrix, point):
        inverse_affine_transform_matrix = cv2.invertAffineTransform(affine_transform_matrix)
        return self.apply_affine_transform_on_point(inverse_affine_transform_matrix, point)

    def apply_affine_transform_on_contour(self, affine_transform_matrix, contour):
        return np.expand_dims(np.asarray([self.apply_affine_transform_on_point(affine_transform_matrix, point) for point in contour]).astype(int),axis = 1)

    def get_blob_to_modify_and_mouse_coordinate(self):
        mouse_coords = self.touches[0]
        frame_index = int(self.visualiser.video_slider.value) #get the current frame from the slider
        blobs_in_frame = self.blobs_in_video[frame_index]
        contours = [getattr(blob, "contour") for blob in blobs_in_frame]
        if self.scale != 1:
            contours = [self.apply_affine_transform_on_contour(self.M, cnt) for cnt in contours]
        mouse_coords = self.fromShowFrameToTexture(mouse_coords)
        if self.scale != 1:
            mouse_coords = self.apply_inverse_affine_transform_on_point(self.M, mouse_coords)
        blob_ind = self.get_clicked_blob(mouse_coords, contours)
        if blob_ind is not None:
            blob_to_modify = blobs_in_frame[blob_ind]
            return blob_to_modify, mouse_coords
        else:
            return None, None

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in showFrame (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        if hasattr(CHOSEN_VIDEO.video, 'resolution_reduction') and  CHOSEN_VIDEO.video.resolution_reduction != 1:
            original_frame_width = int(CHOSEN_VIDEO.video.width * CHOSEN_VIDEO.video.resolution_reduction)
            original_frame_height = int(CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.resolution_reduction)
        else:
            original_frame_width = int(CHOSEN_VIDEO.video.width)
            original_frame_height = int(CHOSEN_VIDEO.video.height)
        actual_frame_width, actual_frame_height = self.visualiser.display_layout.size
        self.offset = self.visualiser.footer.height
        coords[1] = coords[1] - self.offset
        wRatio = abs(original_frame_width / actual_frame_width)
        hRatio = abs(original_frame_height / actual_frame_height)
        ratios = np.asarray([wRatio, hRatio])
        coords =  np.multiply(coords, ratios)
        coords[1] = original_frame_height - coords[1]
        return coords

    @staticmethod
    def get_attributes_from_blobs_in_frame(blobs_in_frame, attributes_to_get):
        return {attr: [getattr(blob, attr) for blob in blobs_in_frame] for attr in attributes_to_get}

    def writeIds(self, frame):
        blobs_in_frame = self.blobs_in_video[int(self.visualiser.video_slider.value)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = self.visualiser.frame
        frame_number = blobs_in_frame[0].frame_number

        for blob in blobs_in_frame:
            cur_id = blob.final_identity
            cur_id_str = str(cur_id)
            roots = ['a-', 'd-', 'c-','i-', 'u-']
            if blob.user_generated_identity is not None:
                root = roots[4]
            elif blob.identity_corrected_closing_gaps is not None and blob.is_an_individual:
                root = roots[3]
            elif blob.identity_corrected_closing_gaps is not None:
                root = roots[2]
            elif blob.identity_corrected_solving_duplication is not None:
                root = roots[1]
            elif not blob.used_for_training:
                root = roots[0]
            else:
                root  = ''
            if isinstance(cur_id, int):
                cur_id_str = root + cur_id_str
                int_centroid = np.asarray(blob.centroid).astype('int')
                cv2.circle(frame, tuple(int_centroid), 2, self.colors[cur_id], -1)
                cv2.putText(frame, cur_id_str,tuple(int_centroid), font, 1, self.colors[cur_id], 3)
                if blob.is_a_crossing or blob.identity_corrected_closing_gaps is not None:
                    bounding_box = blob.bounding_box_in_frame_coordinates
                    cv2.rectangle(frame, bounding_box[0], bounding_box[1], (255, 0, 0) , 2)
            elif isinstance(cur_id, list):
                for c_id, c_centroid in zip(cur_id, blob.interpolated_centroids):
                    c_id_str = root + str(c_id)
                    int_centroid = tuple([int(centroid_coordinate) for centroid_coordinate in c_centroid])
                    cv2.circle(frame, int_centroid, 2, self.colors[c_id], -1)
                    cv2.putText(frame, c_id_str, int_centroid, font, 1, self.colors[c_id], 3)

                self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
                self._keyboard.bind(on_key_down=self._on_keyboard_down)
                if blob.is_a_crossing or blob.identity_corrected_closing_gaps is not None:
                    bounding_box = blob.bounding_box_in_frame_coordinates
                    cv2.rectangle(frame, bounding_box[0], bounding_box[1], (255, 0, 0) , 2)
            elif blob.assigned_identity is None:
                bounding_box = blob.bounding_box_in_frame_coordinates
                cv2.rectangle(frame, bounding_box[0], bounding_box[1], (255, 0, 0) , 2)

        if self.scale != 1:
            self.dst = cv2.warpAffine(frame, self.M, (frame.shape[1], frame.shape[0]))
            buf = cv2.flip(self.dst,0)
            buf = buf.tostring()
        else:
            buf = cv2.flip(frame,0)
            buf = buf.tostring()
        textureFrame = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.visualiser.display_layout.texture = textureFrame

    def on_enter(self,value):
        self.identity_update = int(self.identityInput.text)
        self.overwriteIdentity()
        self.popup.dismiss()
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def propagate_groundtruth_identity_in_individual_fragment(self):
        modified_blob = self.blob_to_modify
        count_past_corrections = 1 #to take into account the modification already done in the current frame
        count_future_corrections = 0
        new_blob_identity = modified_blob.user_generated_identity
        if modified_blob.is_an_individual_in_a_fragment:
            current = modified_blob

            while current.next[0].is_an_individual_in_a_fragment:
                current.next[0]._user_generated_identity = current.user_generated_identity
                current = current.next[0]
                count_future_corrections += 1

            current = modified_blob

            while current.previous[0].is_an_individual_in_a_fragment:
                current.previous[0]._user_generated_identity = current.user_generated_identity
                current = current.previous[0]
                count_past_corrections += 1

        #init and bind keyboard again
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def overwriteIdentity(self):
        # enable buttons to save corrected version and compute the accuracy
        self.save_groundtruth_btn.disabled = False
        self.compute_accuracy_button.disabled = False
        if not self.blob_to_modify.is_a_crossing:
            self.blob_to_modify._user_generated_identity = self.identity_update
            self.propagate_groundtruth_identity_in_individual_fragment()
            self.popup.dismiss()
        self.visualiser.visualise(trackbar_value = int(self.visualiser.video_slider.value), func = self.writeIds)

    def on_press_show_saving(selg, *args):
        self.show_saving()

    def save_groundtruth_list_of_blobs(self, *args):
        self.go_and_save()
        self.popup_saving.dismiss()

    def go_and_save(self):
        self.list_of_blobs.save(CHOSEN_VIDEO.video, path_to_save = self.list_of_blobs_save_path)
        CHOSEN_VIDEO.video.save()

    def modifyIdOpenPopup(self, blob_to_modify):
        self.container = BoxLayout()
        self.blob_to_modify = blob_to_modify
        if blob_to_modify.user_generated_identity is None:
            self.id_to_modify = blob_to_modify.identity
        else:
            self.id_to_modify = blob_to_modify.user_generated_identity
        text = str(self.id_to_modify)
        self.old_id_box = BoxLayout(orientation="vertical")
        self.new_id_box = BoxLayout(orientation="vertical")
        self.selected_label = CustomLabel(text='You selected animal:\n')
        self.selected_label_num = CustomLabel(text=text)
        self.new_id_label = CustomLabel(text='Type the new identity and press enter to confirm\n')
        self.container.add_widget(self.old_id_box)
        self.container.add_widget(self.new_id_box)
        self.old_id_box.add_widget(self.selected_label)
        self.old_id_box.add_widget(self.selected_label_num)
        self.new_id_box.add_widget(self.new_id_label)
        self.identityInput = TextInput(text ='', multiline=False)
        self.new_id_box.add_widget(self.identityInput)
        self.popup = Popup(title='Correcting identity',
            content=self.container,
            size_hint=(.4,.4))
        self.popup.color = (0.,0.,0.,0.)
        self.identityInput.bind(on_text_validate=self.on_enter)
        self.popup.open()

    def show_blob_attributes(self, blob_to_explore):
        self.container = BoxLayout()
        self.blob_to_explore = blob_to_explore
        self.show_attributes_box = BoxLayout(orientation="vertical")
        self.id_label = CustomLabel(text='Assigned identity: ' + str(blob_to_explore.final_identity))
        self.frag_id_label = CustomLabel(text='Fragment identifier: ' + str(blob_to_explore.fragment_identifier))
        self.accumulation_label = CustomLabel(text='Used for training: ' + str(blob_to_explore.used_for_training))
        self.in_a_fragment_label = CustomLabel(text='It is in an individual fragment: ' + str(blob_to_explore.is_in_a_fragment))
        self.individual_label = CustomLabel(text='It is an individual: ' + str(blob_to_explore.is_an_individual))
        self.sure_individual_label = CustomLabel(text='sure individual: ' + str(blob_to_explore.is_a_sure_individual()))
        self.sure_crossing_label = CustomLabel(text='sure crossing: ' + str(blob_to_explore.is_a_sure_crossing()))
        self.ghost_crossing_label = CustomLabel(text='It is a ghost crossing: ' + str(blob_to_explore.is_a_ghost_crossing))
        self.jump_label = CustomLabel(text='It is a jump: ' + str(blob_to_explore.is_a_jump))
        text_centroid_label = str(blob_to_explore.centroid)
        self.centroid_label = CustomLabel(text='Centroid: ' + text_centroid_label)
        self.container.add_widget(self.show_attributes_box)
        widget_list = [self.id_label, self.frag_id_label, self.individual_label,
                        self.sure_individual_label, self.sure_crossing_label,
                        self.accumulation_label, self.in_a_fragment_label,
                        self.ghost_crossing_label, self.jump_label,
                        self.centroid_label]
        [self.show_attributes_box.add_widget(w) for w in widget_list]
        self.popup = Popup(title='Blob attributes',
            content=self.container,
            size_hint=(.4,.4))
        self.popup.color = (0.,0.,0.,0.)
        self.popup.open()

    def on_touch_down(self, touch):
        self.touches = []
        if self.visualiser.display_layout.collide_point(*touch.pos):
            if touch.button =='left':
                self.touches.append(touch.pos)
                self.id_to_modify, self.user_generated_centroids = self.get_blob_to_modify_and_mouse_coordinate()
                if self.id_to_modify is not None:
                    self.modifyIdOpenPopup(self.id_to_modify)
            elif touch.button == 'scrollup':
                self.count_scrollup += 1
                coords = self.fromShowFrameToTexture(touch.pos)
                rows, cols, channels = self.visualiser.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.visualiser.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
            elif touch.button == 'scrolldown':
                coords = self.fromShowFrameToTexture(touch.pos)
                rows,cols, channels = self.visualiser.frame.shape
                self.dst = self.visualiser.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
                self.count_scrollup = 0
                self.scale = 1
            elif touch.button == 'right':
                self.touches.append(touch.pos)
                self.id_to_modify, self.user_generated_centroids = self.get_blob_to_modify_and_mouse_coordinate()
                if self.id_to_modify is not None:
                    self.show_blob_attributes(self.id_to_modify)
        else:
            self.scale = 1
            self.disable_touch_down_outside_collided_widget(touch)

    def disable_touch_down_outside_collided_widget(self, touch):
        return super(Validator, self).on_touch_down(touch)

    def get_groundtruth_path(self):
        groundtruth_path = os.path.join(CHOSEN_VIDEO.video.video_folder, '_groundtruth.npy')
        return groundtruth_path if os.path.isfile(groundtruth_path) else None

    def on_groundtruth_popup_button_press(self, instance):
        if instance.text == "Use pre-existent ground truth":
            self.groundtruth = np.load(self.groundtruth_path).item()
            self.plot_groundtruth_statistics()
            self.popup_start_end_groundtruth.dismiss()
        else:
            self.gt_start_end_container.remove_widget(self.gt_start_end_btn1)
            self.gt_start_end_container.remove_widget(self.gt_start_end_btn2)
            self.gt_start_end_label.text = "Insert the start and ending frame (e.g. 100 - 2050) on which the ground truth has been computed"
            self.gt_start_end_text_input = TextInput(text ='', multiline=False)
            self.gt_start_end_container.add_widget(self.gt_start_end_text_input)
            self.gt_start_end_text_input.bind(on_text_validate = self.on_enter_start_end)

    def create_frame_interval_popup(self):
        self.gt_start_end_container = BoxLayout(orientation = "vertical")
        self.groundtruth_path = self.get_groundtruth_path()
        if self.groundtruth_path is not None:
            if self.save_groundtruth_btn.disabled:
                self.groundtruth = np.load(self.groundtruth_path).item()
                self.plot_groundtruth_statistics()
                return True
            if not self.save_groundtruth_btn.disabled:
                self.gt_start_end_label = CustomLabel(text = "A pre-existent ground truth file has been detected. Do you want to use it to compute the accuracy or use a new one?")
                self.gt_start_end_btn1 = Button(text = "Use pre-existent ground truth")
                self.gt_start_end_btn2 = Button(text = "Generate new ground truth")
                self.gt_start_end_container.add_widget(self.gt_start_end_label)
                self.gt_start_end_container.add_widget(self.gt_start_end_btn1)
                self.gt_start_end_container.add_widget(self.gt_start_end_btn2)
                self.gt_start_end_btn1.bind(on_press = self.on_groundtruth_popup_button_press)
                self.gt_start_end_btn2.bind(on_press = self.on_groundtruth_popup_button_press)
        else:
            if self.save_groundtruth_btn.disabled:
                self.gt_start_end_label = CustomLabel(text = "No pre-existent groundtruth file has been detected. Validate the video to compute a ground truth first.\n\n Need help? To modify a wrong identity click on the badly identified animal and fill the popup. Use the mouse wheel to zoom if necessary.")
                self.gt_start_end_container.add_widget(self.gt_start_end_label)
            else:
                self.gt_start_end_label = CustomLabel(text = "Insert the start and ending frame (e.g. 100 - 2050) on which the ground truth has been computed")
                self.gt_start_end_container.add_widget(self.gt_start_end_label)
                self.gt_start_end_text_input = TextInput(text ='', multiline=False)
                self.gt_start_end_container.add_widget(self.gt_start_end_text_input)
                self.gt_start_end_text_input.bind(on_text_validate = self.on_enter_start_end)

        self.popup_start_end_groundtruth = Popup(title='Groundtruth Accuracy - Frame Interval',
                    content=self.gt_start_end_container,
                    size_hint=(.4,.4))

    def on_enter_start_end(self, value):
        start, end = self.gt_start_end_text_input.text.split('-')
        self.gt_start_frame = int(start)
        self.gt_end_frame = int(end)
        self.generate_groundtruth()
        self.save_groundtruth()
        self.plot_groundtruth_statistics()
        if not self.prevent_open_popup:
            self.popup_start_end_groundtruth.dismiss()
            self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def generate_groundtruth(self):
        self.groundtruth = generate_groundtruth(CHOSEN_VIDEO.video, self.blobs_in_video, self.gt_start_frame, self.gt_end_frame, save_gt = False)

    def save_groundtruth(self):
        self.groundtruth.save()

    def plot_groundtruth_statistics(self):
        blobs_in_video_groundtruth = self.groundtruth.blobs_in_video[self.groundtruth.start:self.groundtruth.end]
        blobs_in_video = self.blobs_in_video[self.groundtruth.start:self.groundtruth.end]
        gt_accuracies, _ = get_accuracy_wrt_groundtruth(CHOSEN_VIDEO.video, blobs_in_video_groundtruth, blobs_in_video)
        if gt_accuracies is not None:
            self.individual_accuracy = gt_accuracies['individual_accuracy']
            self.accuracy = gt_accuracies['accuracy']
            self.plot_final_statistics()
            self.statistics_popup.open()
            CHOSEN_VIDEO.video.gt_start_end = (self.groundtruth.start, self.groundtruth.end)
            CHOSEN_VIDEO.video.gt_accuracy = gt_accuracies
            CHOSEN_VIDEO.video.save()

    def compute_and_save_session_accuracy_wrt_groundtruth_APP(self, *args):
        self.prevent_open_popup = self.create_frame_interval_popup()
        if not self.prevent_open_popup:
            self.popup_start_end_groundtruth.open()

    def plot_final_statistics(self):
        content = BoxLayout()
        self.statistics_popup = Popup(title = "Statistics",
                                    content = content,
                                    size_hint = (.5, .5))
        fig, ax = plt.subplots(1)
        colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals, norm = True, black = True)
        width = .5
        plt.bar(self.individual_accuracy.keys(), self.individual_accuracy.values(), width, color= colors[::-1])
        plt.axhline(self.accuracy, color = 'k', linewidth = .2)
        ax.set_xlabel('individual')
        ax.set_ylabel('Individual accuracy')
        content.add_widget(FigureCanvasKivyAgg(fig))
