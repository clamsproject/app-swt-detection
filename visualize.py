"""visualize.py

Create an HTML file that visualizes the output of the frame classifier.

Usage:

$ python visualize.py -v <VIDEO_FILE> -p <PREDICTIONS_FILE>

    VIDEO_FILE        -  an MP4 video file
    PREDICTIONS_FILE  -  predictions created for VIDEO_FILE (using classify.py)

The styles for the popups are based mostly the second of the two following links:
- https://stackoverflow.com/questions/27004136/position-popup-image-on-mouseover
- https://stackoverflow.com/questions/32153973/how-to-display-popup-on-mouse-over-in-html-using-js-and-css
    
Some things to change here:

- Ignore large stretches where non-other values are below 0.01,
  but keep those that show up in a gold standard if we have one.
- Include information from the TimeFrames that were generated.
- Label names are now hard-coded.

"""


import os, json, argparse
import cv2


# Edit this if we use different labels
LABELS = ('slate', 'chyron', 'credits')
LABELS = ('bars', 'slate', 'chyron', 'credits', 'copy', 'text', 'person')


STYLESHEET = '''
<style>
.none { }
.small { color: #fd0; }
.medium { color: #fa0; }
.large { color: #f60; }
.huge { color: #f20; }
.anchor { color: #666 }
td.popup:hover { z-index: 6; }
td.popup span { position: absolute; left: -9999px; z-index: 6; }
/* Need to change this so that the margin is calculated from the number of columns */
td.popup:hover span { margin-left: 700px; left: 2%; z-index:6; }
</style>
'''


def load_predictions(filename):
    predictions = []
    with open(filename) as fh:
        for (n, tensor, data) in json.load(fh):
            predictions.append((n, data))
    return predictions


def create_frames(video_file: str, predictions: list, frames_dir: str):
    vidcap = cv2.VideoCapture(video_file)
    for milliseconds, scores in predictions:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
        success, image = vidcap.read()
        cv2.imwrite(f"{frames_dir}/frame-{milliseconds:06d}.jpg", image)
        print(milliseconds, success)


def visualize_predictions(predictions: list, labels: list, htmlfile: str, video_file: str):
    with open(htmlfile, 'w') as fh:
        fh.write('<html>\n')
        fh.write(STYLESHEET)
        fh.write('<body>\n')
        fh.write(f'<h2>{video_file}</h2>\n')
        fh.write(f'\n<p>TOTAL PREDICTIONS: {len(predictions)}</p>\n')
        fh.write('<table cellpadding=5 cellspacing=0 border=1>\n')
        fh.write('<tr align="center">\n')
        #for header in ('anchor',) + labels + ('other', 'img'):
        #    fh.write(f'  <td>{header}</td>\n')
        #fh.write('<tr/>\n')
        lines = 0
        for milliseconds, scores in predictions:
            if lines % 20 == 0:
                fh.write('<tr align="center">\n')
                for header in ('anchor',) + labels + ('other', 'img'):
                    fh.write(f'  <td>{header}</td>\n')
                fh.write('<tr/>\n')
            lines += 1
            fh.write('<tr>\n')
            fh.write(f'  <td align="right" class="anchor">{milliseconds}</td>\n')
            for p in scores:
                url = f"frames/frame-{milliseconds:06}.jpg"
                fh.write(f'  <td align="right" class="{get_color_class(p)}">{p:.4f}</td>\n')
            onclick = f"window.open('{url}', '_blank')"
            image = f'<img src="{url}" height="24px">'
            image_popup = f'<img src="{url}">'
            fh.write(
                f'  <td class="popup">\n'
                f'    <a href="#" onClick="{onclick}">{image}</a>\n'
                f'    <span>{image_popup}</span>\n'
                f'  </td>\n')
            fh.write('</tr>\n')
        fh.write('</table>\n')
        fh.write('</body>\n')
        fh.write('</html>\n')


def get_color_class(score: float):
    if score > 0.75:
        return "huge"
    elif score > 0.50:
        return "large"
    elif score > 0.25:
        return "medium"
    elif score > 0.01:
        return "small"
    else:
        return "none"



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", metavar='FILENAME', required=True, help="video file")
    parser.add_argument("-p", metavar='FILENAME', required=True, help="predictions file")
    args = parser.parse_args()

    video_file = args.v
    predictions_file = args.p

    basename = os.path.splitext(os.path.basename(predictions_file))[0]
    outdir = os.path.join('html', basename)
    outdir_frames = os.path.join(outdir, 'frames')
    index_file = os.path.join(outdir, f'index-{"-".join(LABELS)}.html')
    os.makedirs(outdir_frames, exist_ok=True)

    predictions = load_predictions(predictions_file)
    #create_frames(video_file, predictions, outdir_frames)
    visualize_predictions(predictions, LABELS, index_file, video_file)
