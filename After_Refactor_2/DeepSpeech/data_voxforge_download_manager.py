from six.moves import urllib
import os
import shutil
import tempfile
import tarfile

class DownloadManager:
    def __init__(self, url, target_dir):
        self.url = url
        self.target_dir = target_dir

    def download_and_extract(self, filename, recording_name):
        """
        Downloads and extracts a sample from VoxForge and puts the wav and txt files into :target_folder.
        """
        wav_dir = os.path.join(self.target_dir, "wav")
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
        txt_dir = os.path.join(self.target_dir, "txt")
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        # check if sample is processed
        filename_set = set(['_'.join(wav_file.split('_')[:-1]) for wav_file in os.listdir(wav_dir)])
        if recording_name in filename_set:
            return

        request = urllib.request.Request(self.url + filename)
        response = urllib.request.urlopen(request)
        content = response.read()
        response.close()
        with tempfile.NamedTemporaryFile(suffix=".tgz", mode='wb') as target_tgz:
            target_tgz.write(content)
            target_tgz.flush()
            dirpath = tempfile.mkdtemp()

            tar = tarfile.open(target_tgz.name)
            tar.extractall(dirpath)
            tar.close()

            recordings_type, recordings_dir = get_recordings_dir(dirpath, recording_name)
            tgz_prompt_file = os.path.join(dirpath, recording_name, "etc", "PROMPTS")

            if os.path.exists(recordings_dir) and os.path.exists(tgz_prompt_file):
                transcriptions = open(tgz_prompt_file).read().strip().split("\n")
                transcriptions = {t.split()[0]: " ".join(t.split()[1:]) for t in transcriptions}
                for wav_file in os.listdir(recordings_dir):
                    recording_id = wav_file.split('.{}'.format(recordings_type))[0]
                    transcription_key = recording_name + "/mfc/" + recording_id
                    if transcription_key not in transcriptions:
                        continue
                    utterance = transcriptions[transcription_key]

                    target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(recording_name, recording_id))
                    target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(recording_name, recording_id))
                    with open(target_txt_file, "w") as file:
                        file.write(utterance)
                    original_wav_file = os.path.join(recordings_dir, wav_file)
                    subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(original_wav_file, str(args.sample_rate),
                                                                          target_wav_file)], shell=True)

            shutil.rmtree(dirpath)