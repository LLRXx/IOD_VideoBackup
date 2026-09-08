import pickle
import os
import cv2
import numpy as np

label_class = ['LeakedGas']

instance_color = ['yellow', 'lime', 'MediumVioletRed', 'Cyan', 'DarkOrange', 'Red', 'Navy', 'Indigo', 'RoyalBlue']


def takeLast(elem):
    return elem[-1]

def pkl_decode(opt):
    print('build finish, decode detection results', flush=True)
    with open(os.path.join(opt.inference_dir, 'tubes.pkl'), 'rb') as fid:
        pkl = pickle.load(fid)

    bbox_dict = {}
    tube_id = 0
    for label in pkl.keys():
        out = pkl[label]
        if len(out) > 0:
            out.sort(key=takeLast, reverse=True)
            for tube in out:
                tube_score = tube[1]
                if tube_score > opt.tube_vis_th:
                    label_name = label_class[label]
                    for frame in range(tube[0].shape[0]):
                        frame_score = tube[0][frame][5]
                        if frame_score > opt.frame_vis_th:
                            fid = tube[0][frame][0]
                            x1, y1, x2, y2 = tube[0][frame][1], tube[0][frame][2], tube[0][frame][3], tube[0][frame][4]
                            if fid not in bbox_dict:
                                bbox_dict[fid] = []
                                bbox_dict[fid].append([x1, y1, x2, y2, frame_score, label_name, tube_id])
                            else:
                                bbox_dict[fid].append([x1, y1, x2, y2, frame_score, label_name, tube_id])
                    tube_id += 1
    return bbox_dict#{frame:[x1, y1, x2, y2, frame_score, label_name, tube_id]}


def vis_bbox(opt, inference_dir, bbox_dict, instance_level=False):
    print('draw bboxes on each frame', flush=True)
    if not os.path.isdir(os.path.dirname('tmp')):
        os.system("mkdir -p tmp")

    # Draw directly with OpenCV instead of Matplotlib. The original
    # Rectangle-based implementation is incompatible with some newer
    # NumPy/Matplotlib combinations and fails inside Path.iter_segments.
    im_list = os.listdir(inference_dir)
    im_list.sort()
    for pic in im_list:
        if pic.endswith('.jpg') or pic.endswith('.png'):
            image_path = os.path.join(inference_dir, pic)
            im_data = cv2.imread(image_path)
            if im_data is None:
                continue

            fid = int(pic.split('.')[0])
            bbox_list = bbox_dict.get(fid, [])
            for bbox in bbox_list:
                coords = np.asarray(bbox[:4], dtype=np.float64).reshape(-1)
                if coords.size != 4 or not np.isfinite(coords).all():
                    continue
                x1, y1, x2, y2 = coords.tolist()
                score = float(bbox[4])

                if score < opt.simple_th and opt.SimpleFrameProcess:
                    continue

                # OpenCV uses BGR colors; yellow is (0, 255, 255).
                color = (0, 255, 255)
                if instance_level and int(bbox[6]) < len(instance_color):
                    color_map = {
                        'yellow': (0, 255, 255),
                        'lime': (0, 255, 0),
                        'MediumVioletRed': (199, 21, 133),
                        'Cyan': (255, 255, 0),
                        'DarkOrange': (0, 140, 255),
                        'Red': (0, 0, 255),
                        'Navy': (128, 0, 0),
                        'Indigo': (130, 0, 75),
                        'RoyalBlue': (225, 105, 65),
                    }
                    color = color_map.get(instance_color[int(bbox[6])], color)

                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(im_data, pt1, pt2, color, 2)

                text = bbox[5] + ', ' + "%.2f" % score
                text_y = max(15, pt1[1] - 8)
                cv2.putText(im_data, text, (pt1[0], text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)

            cv2.imwrite(os.path.join('tmp', pic), im_data)


def video2frames(opt):
    # if opt.pre_extracted_brox_flow:
    #     pre_extracted_frames(opt)
    # else:
    print('start extracting frames')
    vidcap = cv2.VideoCapture(os.path.join(opt.DATA_ROOT, opt.vname))
    success, image = vidcap.read()
    fid = 1
    while success:
        cv2.imwrite(os.path.join(opt.inference_dir, 'VideoFrames', '{:0>5}.jpg'.format(fid)), image)
        fid = fid + 1
        success, image = vidcap.read()


# def pre_extracted_frames(opt):
#     print('moving frames(JPG, Flow)')
#     os.system("cp " + os.path.join(opt.IMAGE_ROOT, 'rgb-images', opt.vname, '*') + " " + os.path.join(opt.inference_dir, 'rgb'))
#     os.system("cp " + os.path.join(opt.IMAGE_ROOT, 'brox-images', opt.vname, '*') + " " + os.path.join(opt.inference_dir, 'flow'))


def rgb2avi(inference_dir,video_name = 'result_video.avi'):
    print('convert .JPG to .AVI', flush=True)
    fps = 25
    height, width, _ = cv2.imread('tmp/00001.jpg').shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(inference_dir + '/' + video_name, fourcc, fps, size)

    filelist = os.listdir('tmp')
    filelist.sort()
    for pic in filelist:
        if pic.endswith('.jpg') or pic.endswith('.png'):
            video.write(cv2.imread(os.path.join('tmp', pic)))

    video.release()
    cv2.destroyAllWindows()
    # os.system("rm -rf tmp")
    # os.system("rm -rf " + inference_dir + "/rgb")



def rgb2gif(opt):
    print('convert .JPG to .GIF', flush=True)
    import imageio
    GIF = []
    filelist = os.listdir('tmp')
    filelist.sort()
    for pic in filelist:
        if pic.endswith('.jpg') or pic.endswith('.png'):
            pic = cv2.imread(os.path.join('tmp', pic))[:, :, ::-1]
            GIF.append(pic)
    imageio.mimsave(opt.inference_dir + '/' + opt.vname[:-4] + '.gif', GIF, duration=0.04)  # the lower duration, the quicker gif speed
