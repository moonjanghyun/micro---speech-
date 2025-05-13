
// TensorFlow Lite for Microcontrollers를 이용해 아두이노에서 호출어를 감지하는 메인 프로그램.
// Main program for keyword spotting on Arduino using TensorFlow Lite for Microcontrollers.

// 오디오 입력 >> 특징 추출 >> 입력 텐서 복사 >> 모델 추론 >> 출력 해석 >> 명령어 인식 및 동작. 


#include <TensorFlowLite.h>   // TFLite Micro 메인 라이브러리  (TFLite Micro main library)

// <모델 처리 관련 헤더>  (Headers related to model processing)

#include "audio_provider.h"               // 오디오 입력 처리 (Audio input processing)
#include "command_responder.h"           // 명령 처리 결과 응답 (Responding to command processing results)
#include "feature_provider.h"            // 오디오 특징 추출 (Audio feature extraction)
#include "main_functions.h"              // 메인 함수 정의 (Main function definitions)
#include "micro_features_micro_model_settings.h"  // 모델 파라미터 설정 (Model parameter settings)
#include "micro_features_model.h"        // 모델 데이터 자체 (The model data itself)
#include "recognize_commands.h"          // 명령어 인식 로직 (Command recognition logic)
#include "tensorflow/lite/micro/micro_interpreter.h" // TFLite 인터프리터 (TFLite interpreter)
#include "tensorflow/lite/micro/micro_log.h"         // 디버깅용 로그 (Logging for debugging)
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // 연산자 등록기 (Operator resolver)
#include "tensorflow/lite/micro/system_setup.h"      // 하드웨어 설정 (Hardware system setup)
#include "tensorflow/lite/schema/schema_generated.h" // TFLite 모델 스키마 정의 (TFLite model schema definition)


#undef PROFILE_MICRO_SPEECH

// 전역 변수 및 텐서 초기화.
// Global variables and tensor initialization.
namespace {

// 전역 변수 선언
// Declare global variables
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// 모델 실행에 필요한 메모리 공간(tensor_arena) 생성
// Allocate memory arena for model execution
constexpr int kTensorArenaSize = 10 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// 특징 벡터 저장 공간 및 입력 버퍼 포인터
// Buffer for storing feature vectors and pointer for input data
int8_t feature_buffer[kFeatureElementCount]; //오디오 특징을 저장할 버퍼
// Buffer to store audio features
int8_t* model_input_buffer = nullptr; // 입력 데이터를 위한 버퍼 
// Buffer pointer for input data
}


// 초기화 설정 (TensorFlow Lite Micro 모델을 아두이노에서 실행하기 위해)
// Initialization for running TensorFlow Lite Micro model on Arduino
void setup() {
  tflite::InitializeTarget(); // 하드웨어 초기화
  // Initialize target hardware

  model = tflite::GetModel(g_model); // 모델 로드 및 버전 체크
  // Load the model and check version compatibility
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("버전 불일치.");
    // Version mismatch
    return;
  }

  // 모델에서 사용할 연산(op) 목록을 등록하는 객체를 생성. (최대 4개)
  // Create op resolver and register required operations (up to 4)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  
  // DepthwiseConv2D 연산을 등록 (모바일/임베디드 모델에서 자주 사용되는 경량 합성곱) + 등록 실패 시 setup()을 조기 종료함
  // Register DepthwiseConv2D operation (commonly used in mobile/embedded models)
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  // 완전 연결층(Fully Connected layer) 추가
  // Add FullyConnected layer
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  // Softmax 연산 추가 (분류 모델에서 확률값으로 바꿀 때 사용)
  // Add Softmax operation (for converting logits to probabilities in classification models)
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  // Reshape 연산 추가 (텐서 모양 변경용)
  // Add Reshape operation (for reshaping tensors)
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // 모델을 실행할 인터프리터 생성.
  // Create interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 텐서 메모리 할당. (모델에 필요한 텐서 공간을 tensor_arena에서 할당)
  // Allocate tensors from tensor_arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // 입력 텐서 확인. (모델의 입력 텐서가 예상한 모양인지 검증)
  // Verify model input tensor shape and type
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }

  // 입력 버퍼 포인터 지정. (오디오 특징 데이터를 이 버퍼에 복사)
  // Set input buffer pointer (audio features will be copied into this)
  model_input_buffer = model_input->data.int8;

  // FeatureProvider 초기화. (입력된 오디오를 분석해, 모델에 넣을 수 있는 형태로 변환해주는 객체)
  // Initialize FeatureProvider (converts raw audio to features for the model)
  static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer); // 오디오 스펙트로그램을 생성하는 객체 초기화
  // Create FeatureProvider instance to generate audio spectrogram
  feature_provider = &static_feature_provider; // feature_buffer는 생성된 특징(특성)들이 저장될 공간
  // feature_buffer stores extracted features

  // Recognizer 초기화. (모델 출력값으로부터 명령어(wake word 등)를 인식하는 클래스)
  // Initialize RecognizeCommands (interprets model output to detect commands)
  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  // 타임스탬프 초기화 (이전 오디오 프레임의 시간을 기록)
  // Initialize timestamp for tracking audio frame timing
  previous_time = 0;

  // 오디오 녹음 시작 / 마이크 입력 시작.
  // Start audio recording / microphone input
  TfLiteStatus init_status = InitAudioRecording();
  if (init_status != kTfLiteOk) {
    MicroPrintf("Unable to initialize audio");
    return;
  }
  
  // 초기화 완료 메시지 출력.
  // Print initialization complete message
  MicroPrintf("Initialization complete");
}


// 반복문 - 음성 인식 수행. (지속적으로 오디오 데이터를 가져오고, 추론을 실행, 명령어에 응답.)
// Main loop - perform continuous audio inference and respond to commands
void loop() {
#ifdef PROFILE_MICRO_SPEECH
  const uint32_t prof_start = millis();
  static uint32_t prof_count = 0;
  static uint32_t prof_sum = 0;
  static uint32_t prof_min = std::numeric_limits<uint32_t>::max();
  static uint32_t prof_max = 0;
#endif 

  // 새로운 오디오 데이터를 기반으로 특징 추출
  // Extract features from new audio data
  const int32_t current_time = LatestAudioTimestamp(); // 가장 최신 오디오 타임스탬프 가져오기
  // Get latest audio timestamp
  int how_many_new_slices = 0;                         // 오디오 슬라이스 수
  // Number of new audio slices
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData( //오디오 특징 데이터 추출
      previous_time, current_time, &how_many_new_slices);
  // Extract audio feature data
  
  // 특징 추출 실패라면 오류 메시지 출력. 
  // Print error if feature extraction fails
  if (feature_status != kTfLiteOk) {
    MicroPrintf("Feature generation failed");
    return;
  }
  previous_time += how_many_new_slices * kFeatureSliceStrideMs; // previous time 업데이트 
  // Update previous timestamp
  
  // 새로운 오디오 슬라이스 수가 없다면(0) 스킵. 
  // Skip if no new slices are available
  if (how_many_new_slices == 0) {
    return;
  }

  // 특징 버퍼를 입력으로 복사.
  // Copy feature buffer to model input
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // 입력에 대해 모델을 실행하고 성공 여부 확인.
  // Run inference on input and check success
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) { // 실패라면 오류 출력 후 스킵.
    // Print error and skip if invocation fails
    MicroPrintf("Invoke failed");
    return;
  }

  // 출력 텐서에 대한 포인터 가져오기.
  // Get pointer to output tensor
  TfLiteTensor* output = interpreter->output(0); // 첫 번째 출력 텐서 가져오기
  // Get first output tensor

  // 추론 결과를 바탕으로 명령어가 인식되었는지 판단.
  // Determine if a command was recognized from inference result
  const char* found_command = nullptr; // 인식된 명령어 문자열
  // Recognized command string
  uint8_t score = 0; // 인식된 명령어의 신뢰도 점수
  // Confidence score of recognized command
  bool is_new_command = false; // 새로윤 명령어가 인식되었는지 여부 나타냄
  // Indicates whether it's a new command

  // 최신 추론 결과 처리.
  // Process latest inference results
  TfLiteStatus process_status = recognizer->ProcessLatestResults( 
      output, current_time, &found_command, &score, &is_new_command);
  // Analyze output and determine command

  // 결과 처리 오류 발생 시, 오류 출력 후 loop 중단.
  // Print error and stop loop if result processing fails
  if (process_status != kTfLiteOk) {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }

  RespondToCommand(current_time, found_command, score, is_new_command); // 감지된 명령에 따라 동작 수행.
  // Perform action based on detected command

  // 옵션 : 성능 측정용 (loop() 함수의 처리 시간을 주기적으로 측정 및 출력하여 시스템 성능을 모니터링하는 데 사용)
  // Optional: For performance profiling (used to periodically measure and print the execution time of the loop() function to monitor system performance)
#ifdef PROFILE_MICRO_SPEECH
  const uint32_t prof_end = millis();
  if (++prof_count > 10) {
    uint32_t elapsed = prof_end - prof_start;
    prof_sum += elapsed;
    if (elapsed < prof_min) {
      prof_min = elapsed;
    }
    if (elapsed > prof_max) {
      prof_max = elapsed;
    }
    if (prof_count % 300 == 0) {
      MicroPrintf("## time: min %dms  max %dms  avg %dms", prof_min, prof_max,
                  prof_sum / prof_count);
    }
  }
#endif 
}
