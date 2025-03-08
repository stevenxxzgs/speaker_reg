import os
import time
import queue
import pickle
import threading
import collections
import numpy as np
import webrtcvad
import pyaudio
import wave
import librosa
from collections import deque
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr

class UnsupervisedRealtimeSpeechRecognition:
    """无监督实时语音识别和说话人聚类系统"""
    
    def __init__(self):
        # 音频参数
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 采样率必须是8000, 16000, 32000或48000
        self.chunk_duration_ms = 30  # 每个缓冲区的持续时间(毫秒)
        self.chunk_size = int(self.rate * self.chunk_duration_ms / 1000)  # 缓冲区大小
        self.vad_level = 3  # VAD敏感度 (0-3)
        
        # 初始化VAD
        self.vad = webrtcvad.Vad(self.vad_level)
        
        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()
        
        # 初始化语音识别器
        self.recognizer = sr.Recognizer()
        self.language = "zh-CN"  # 默认中文
        
        # 说话人特征和聚类
        self.speaker_features = []  # 所有说话人特征
        self.speaker_labels = []    # 对应的说话人标签
        self.min_similarity = 0.85  # 聚类相似度阈值
        self.scaler = StandardScaler()
        self.speaker_count = 0      # 当前发现的说话人数量
        
        # 缓冲区和队列
        self.frames = deque(maxlen=500)  # 约10秒的音频
        self.voiced_frames = []
        self.is_speaking = False
        self.speech_start_time = 0
        self.min_speech_duration = 0.5  # 最小语音持续时间(秒)
        self.speech_timeout = 0.8  # 语音超时时间(秒)
        self.last_voice_time = 0
        
        # 结果队列
        self.result_queue = queue.Queue()
        
        # 控制标志
        self.running = False
        
        # 保存说话人特征的文件
        self.features_path = "speaker_features.pkl"
        self.load_features()
    
    def get_speech_features(self, frames):
        """从语音帧中提取说话人特征"""
        # 将帧转换为字节数组
        frame_bytes = b''.join(frames)
        
        # 写入临时WAV文件
        temp_file = "temp_speech.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(frame_bytes)
        
        # 使用librosa提取特征
        try:
            y, sr = librosa.load(temp_file, sr=None)
            
            # 提取MFCC特征 (主要用于说话人识别)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfccs.T, axis=0)
            mfcc_var = np.var(mfccs.T, axis=0)
            
            # 基音特征
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # 谱质心
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            cent_mean = np.mean(cent)
            
            # 合并特征
            features = np.concatenate([mfcc_mean, mfcc_var, [pitch_mean, cent_mean]])
            
            return features
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def recognize_text_from_frames(self, frames):
        """从帧识别文本"""
        # 将帧转换为字节数组
        frame_bytes = b''.join(frames)
        
        # 写入临时WAV文件
        temp_file = "temp_speech.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(frame_bytes)
        
        # 使用SpeechRecognition识别文本
        try:
            with sr.AudioFile(temp_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
        except Exception as e:
            print(f"语音识别错误: {e}")
            return None
    
    def identify_speaker(self, features):
        """无监督说话人识别"""
        if len(self.speaker_features) == 0:
            # 第一个说话人
            self.speaker_count += 1
            self.speaker_features.append(features)
            self.speaker_labels.append(self.speaker_count)
            self.save_features()
            return f"说话人{self.speaker_count}"
        
        # 计算与现有说话人的相似度
        similarities = []
        for existing_features in self.speaker_features:
            # 计算余弦相似度
            similarity = cosine_similarity([features], [existing_features])[0][0]
            similarities.append(similarity)
        
        # 找出最大相似度及其索引
        max_similarity = max(similarities)
        max_similarity_idx = similarities.index(max_similarity)
        
        if max_similarity >= self.min_similarity:
            # 与现有说话人相似
            speaker_label = self.speaker_labels[max_similarity_idx]
            
            # 更新特征 (加权平均，给予现有特征更高权重)
            self.speaker_features[max_similarity_idx] = 0.7 * self.speaker_features[max_similarity_idx] + 0.3 * features
            self.save_features()
            
            return f"说话人{speaker_label}"
        else:
            # 新说话人
            self.speaker_count += 1
            self.speaker_features.append(features)
            self.speaker_labels.append(self.speaker_count)
            self.save_features()
            
            return f"说话人{self.speaker_count}"
    
    def save_features(self):
        """保存说话人特征到文件"""
        with open(self.features_path, 'wb') as f:
            data = {
                'features': self.speaker_features,
                'labels': self.speaker_labels,
                'count': self.speaker_count
            }
            pickle.dump(data, f)
    
    def load_features(self):
        """从文件加载说话人特征"""
        if os.path.exists(self.features_path):
            try:
                with open(self.features_path, 'rb') as f:
                    data = pickle.load(f)
                    self.speaker_features = data['features']
                    self.speaker_labels = data['labels']
                    self.speaker_count = data['count']
                print(f"已加载 {len(self.speaker_features)} 个说话人特征")
            except Exception as e:
                print(f"加载特征失败: {e}")
                self.speaker_features = []
                self.speaker_labels = []
                self.speaker_count = 0
    
    def audio_processing_thread(self):
        """音频处理线程"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        try:
            while self.running:
                # 读取音频块
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
                
                # 检查是否有语音
                is_speech = self.vad.is_speech(data, self.rate)
                
                current_time = time.time()
                
                if is_speech:
                    self.last_voice_time = current_time
                    
                    if not self.is_speaking:
                        # 开始新的语音段
                        self.is_speaking = True
                        self.speech_start_time = current_time
                        self.voiced_frames = []
                    
                    # 添加到当前语音段
                    self.voiced_frames.append(data)
                
                elif self.is_speaking:
                    # 检查是否超过语音超时时间
                    if current_time - self.last_voice_time > self.speech_timeout:
                        # 如果语音段足够长，处理它
                        speech_duration = current_time - self.speech_start_time
                        if speech_duration >= self.min_speech_duration and len(self.voiced_frames) > 15:
                            # 处理语音段 - 启动识别线程
                            speech_frames = self.voiced_frames.copy()
                            
                            # 语音处理线程
                            threading.Thread(
                                target=self.process_speech,
                                args=(speech_frames,),
                                daemon=True
                            ).start()
                        
                        # 重置语音段
                        self.is_speaking = False
                        self.voiced_frames = []
        finally:
            stream.stop_stream()
            stream.close()
    
    def process_speech(self, frames):
        """处理语音段 - 同时进行文本识别和说话人识别"""
        # 文本识别
        text = self.recognize_text_from_frames(frames)
        
        # 说话人特征提取和识别
        features = self.get_speech_features(frames)
        if features is not None:
            speaker = self.identify_speaker(features)
        else:
            speaker = "未知说话人"
        
        # 将结果放入队列
        if text:
            self.result_queue.put((speaker, text))
    
    def display_results_thread(self):
        """结果显示线程"""
        # 保存最后10条记录，用于界面显示
        last_results = collections.deque(maxlen=10)
        
        while self.running:
            # 检查是否有新结果
            try:
                while not self.result_queue.empty():
                    speaker, text = self.result_queue.get(block=False)
                    timestamp = time.strftime("%H:%M:%S")
                    result = (timestamp, speaker, text)
                    last_results.append(result)
                    
                    # 清屏并显示
                    print("\033c", end="")  # 清屏
                    
                    print("=" * 60)
                    print("实时语音转文字与说话人识别系统")
                    print("=" * 60)
                    print(f"已识别说话人数量: {self.speaker_count}")
                    print("-" * 60)
                    
                    # 显示最近结果
                    for ts, spk, txt in last_results:
                        print(f"[{ts}] {spk}: {txt}")
                    
                    print("-" * 60)
                    print("按 Ctrl+C 停止")
            except queue.Empty:
                pass
            
            time.sleep(0.1)  # 降低CPU使用率
    
    def clear_speakers(self):
        """清除所有说话人数据"""
        self.speaker_features = []
        self.speaker_labels = []
        self.speaker_count = 0
        
        if os.path.exists(self.features_path):
            os.remove(self.features_path)
        
        print("已清除所有说话人数据")
    
    def start(self):
        """启动实时识别系统"""
        if self.running:
            print("系统已经在运行中")
            return
        
        self.running = True
        
        # 启动音频处理线程
        threading.Thread(target=self.audio_processing_thread, daemon=True).start()
        
        # 启动结果显示线程
        threading.Thread(target=self.display_results_thread, daemon=True).start()
        
        print("无监督实时语音识别系统已启动")
        print("开始说话后会自动检测并显示结果")
        print("按 Ctrl+C 停止")
    
    def stop(self):
        """停止实时识别系统"""
        self.running = False
        print("\n系统已停止")
    
    def run_demo(self):
        """运行演示程序"""
        print("=" * 60)
        print("无监督实时语音转文字与说话人识别系统")
        print("=" * 60)
        
        while True:
            print("\n选择操作:")
            print("1. 启动实时识别 (自动识别不同说话人)")
            print("2. 清除说话人数据")
            print("3. 调整识别灵敏度")
            print("4. 退出")
            
            choice = input("请输入选项 (1-4): ")
            
            if choice == '1':
                try:
                    self.start()
                    input("\n按回车键停止...\n")
                    self.stop()
                except KeyboardInterrupt:
                    self.stop()
            
            elif choice == '2':
                self.clear_speakers()
            
            elif choice == '3':
                print("\n当前设置:")
                print(f"VAD灵敏度: {self.vad_level}/3")
                print(f"说话人相似度阈值: {self.min_similarity}")
                print(f"最小语音持续时间: {self.min_speech_duration}秒")
                print(f"语音超时时间: {self.speech_timeout}秒")
                
                print("\n调整哪个参数?")
                print("1. VAD灵敏度 (越高越敏感)")
                print("2. 说话人相似度阈值 (越高区分度越强)")
                print("3. 最小语音持续时间 (太短的语音会被忽略)")
                print("4. 语音超时时间 (语音间隙多久算一句话结束)")
                
                param_choice = input("请选择 (1-4): ")
                
                if param_choice == '1':
                    value = int(input("输入新的VAD灵敏度 (0-3): "))
                    if 0 <= value <= 3:
                        self.vad_level = value
                        self.vad = webrtcvad.Vad(self.vad_level)
                        print(f"VAD灵敏度已设置为 {value}")
                    else:
                        print("无效的值，必须在0-3范围内")
                
                elif param_choice == '2':
                    value = float(input("输入新的相似度阈值 (0.1-1.0): "))
                    if 0.1 <= value <= 1.0:
                        self.min_similarity = value
                        print(f"相似度阈值已设置为 {value}")
                    else:
                        print("无效的值，必须在0.1-1.0范围内")
                
                elif param_choice == '3':
                    value = float(input("输入新的最小语音持续时间 (秒): "))
                    if value > 0:
                        self.min_speech_duration = value
                        print(f"最小语音持续时间已设置为 {value}秒")
                    else:
                        print("无效的值，必须大于0")
                
                elif param_choice == '4':
                    value = float(input("输入新的语音超时时间 (秒): "))
                    if value > 0:
                        self.speech_timeout = value
                        print(f"语音超时时间已设置为 {value}秒")
                    else:
                        print("无效的值，必须大于0")
            
            elif choice == '4':
                print("感谢使用，再见!")
                break
            
            else:
                print("无效的选项，请重新输入")


if __name__ == "__main__":
    # 创建系统实例并运行
    system = UnsupervisedRealtimeSpeechRecognition()
    try:
        system.run_demo()
    except KeyboardInterrupt:
        if system.running:
            system.stop()
        print("\n程序已退出")