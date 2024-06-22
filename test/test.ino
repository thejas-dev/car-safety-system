int motor1Pin1 = 27; 
int motor1Pin2 = 26; 
int motor2Pin1 = 25; 
int motor2Pin2 = 33;
int enable2Pin = 32; 

#include "DHT.h"
#define DHTPIN 19
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define BLYNK_TEMPLATE_ID           "TMPLrVaJ9uNa"
#define BLYNK_TEMPLATE_NAME         "Quickstart Device"
#define BLYNK_AUTH_TOKEN            "ZQ5E0FZtXGWLgG6IyKrmRMKYjetQS3gB"

/* Comment this out to disable prints and save space */
//#define BLYNK_PRINT Serial  

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

// Your WiFi credentials.
// Set password to "" for open networks.
char ssid[] = "Airtel_Udaya";
char pass[] = "1234567890";

BlynkTimer timer;


int speed = 0;
#define motor 14
int dataSent = 0;
int faceNotMatching = 0;
int NoMatching = 0;
int IncorrectID = 0;
int engineOn = 0;
int acceleratorOnline = 0;
int drowsy = 0;
int tempWarn = 0;
String data;
float t;
float h;

TaskHandle_t Task1;

void myTimerEvent()
{
  // You can send any value at any time.
  // Please don't send more that 10 values per second.
  Blynk.virtualWrite(V1,speed);
  Blynk.virtualWrite(V5,t);
}

BLYNK_WRITE(V2){
  int value = param.asInt();
  Serial.println(value);
  // Update state
  Blynk.virtualWrite(V2, value);
  if(value == 1){
    engineOn = 1;
    speed = 20;
    analogWrite(motor,speed);
    Blynk.virtualWrite(V1,speed);
    Blynk.virtualWrite(V7,1);
    Blynk.virtualWrite(V4,"Ya! got some power");
    digitalWrite(12,HIGH);
  }else{
    engineOn = 0;
    speed = 0;
    analogWrite(motor,0);
    Blynk.virtualWrite(V1,0);
    Blynk.virtualWrite(V7,0);
    Blynk.virtualWrite(V4,"Good Night Buddy!");
    digitalWrite(12,LOW);
  }
}

BLYNK_WRITE(V6){
  int value = param.asInt();
  if(value == 1){
    Serial.println("Change");
    Blynk.virtualWrite(V4,"Scanning new face!");
  }
}

BLYNK_WRITE(V9){
  int value = param.asInt();
  if(value == 1){
    acceleratorOnline = 1;
  }else{
    acceleratorOnline = 0;
  }
}

BLYNK_WRITE(V8){
  int value = param.asInt();
  if(value == 1){
    if(t>25 && t<35){
      analogWrite(enable2Pin,120);
      Blynk.virtualWrite(V4,"Coolant On!");
    }else if(t>=35 && t<50){
      analogWrite(enable2Pin,180);
      Blynk.virtualWrite(V4,"Coolant On!");
    }else if(t>=50 && t<80){
      analogWrite(enable2Pin,240);
      Blynk.virtualWrite(V4,"Coolant On!");
    }else if(t>=80){
      analogWrite(enable2Pin,255);
      Blynk.virtualWrite(V4,"Coolant On!");
    }else{
      analogWrite(enable2Pin,100);
      Blynk.virtualWrite(V4,"Normal temperature!");
      
    }
  }
}

void loop2(void * parameter){
  // pinMode(14,INPUT);
  // pinMode(13,INPUT);
  // pinMode(26,INPUT);
  for(;;){
  int accelerator;
  if(engineOn == 0 || drowsy == 1){
    accelerator = 0;
  } else {
    accelerator = acceleratorOnline;
    // digitalRead(14);
  }  
  int motorOn = 0;
  //digitalRead(13);
  int motorOff = 0;
  //digitalRead(26);
  int count = 0;
  while(motorOn == 1 & engineOn == 0){
    int motorPower = digitalRead(13);
    Blynk.virtualWrite(V6,1);
    if(motorPower == 0){
      break;
    }
    count++;
    if(count > 4){
      engineOn = 1;
      speed = 20;
      analogWrite(motor,speed);
      Blynk.virtualWrite(V1,speed);
      Blynk.virtualWrite(V2,1);
      Blynk.virtualWrite(V7,1);
      Blynk.virtualWrite(V6,0);
      Blynk.virtualWrite(V4,"Ya! got some power");
      digitalWrite(12,HIGH);
      break;
    }
    delay(400);
  }
  while(motorOff == 1 & engineOn == 1 & speed == 20){
    int motorPower = digitalRead(26);
    Blynk.virtualWrite(V6,1);
    if(motorPower == 0){
      break;
    }
    count++;
    if(count > 4){
      engineOn = 0;
      speed = 0;
      analogWrite(motor,0);
      Blynk.virtualWrite(V1,0);
      Blynk.virtualWrite(V2,0);
      Blynk.virtualWrite(V7,0);
      Blynk.virtualWrite(V6,0);
      Blynk.virtualWrite(V4,"Good Night Buddy!");
      digitalWrite(12,LOW);
      break;
    }
    delay(400);
  }
  if(accelerator == 1 && dataSent == 0){
    dataSent = 1;
    Serial.println("Check Data");    
  }
  if(accelerator == 1 && speed<255 && faceNotMatching == 0 && IncorrectID == 0){
    speed++;
    analogWrite(motor,speed);
  }else if((faceNotMatching == 1 || NoMatching == 1 || IncorrectID == 1) && speed > 20){
    speed--;
    analogWrite(motor,speed);
  }else if(accelerator == 0 && speed > 20){
    dataSent = 0;
    speed--;
    analogWrite(motor,speed);
  }else if(accelerator == 1 && speed > 20){
    speed--;
    analogWrite(motor,speed);
  }
  if(accelerator == 0){
    dataSent = 0;
  }
  delay(10);
  }
}

void setup() {
  // put your setup code here, to run once:
  
  pinMode(motor,OUTPUT);
  pinMode(12,OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(motor1Pin1,OUTPUT);
  pinMode(motor1Pin2,OUTPUT);
  pinMode(motor2Pin1,OUTPUT);
  pinMode(motor2Pin2,OUTPUT);
  pinMode(enable2Pin,OUTPUT);  
  Serial.begin(115200);
  dht.begin();
  xTaskCreatePinnedToCore(
    loop2,
    "Task1",
    10000,
    NULL,
    1,
    &Task1,
    0);
  analogWrite(motor,0);
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  // You can also specify server:
  //Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass, "blynk.cloud", 80);
  //Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass, IPAddress(192,168,1,100), 8080);

  // Setup a function to be called every second
  timer.setInterval(100L, myTimerEvent);
  Blynk.virtualWrite(V3,"Happy Ride!");
  Blynk.virtualWrite(V4,"Drive safe, its not GTA.");
  digitalWrite(12,LOW);
  digitalWrite(motor1Pin1,HIGH);
  digitalWrite(motor1Pin2,LOW);
  digitalWrite(motor2Pin1,HIGH);
  digitalWrite(motor2Pin2,LOW);
}

void loop() {
  // put your main code here, to run repeatedly:
  Blynk.run();
  timer.run();
  t = dht.readTemperature();
  if(t>45 && t<70){
    analogWrite(enable2Pin,100);
    Blynk.virtualWrite(V3,"Warning! High temperature");
    Blynk.virtualWrite(V4,"Activating ventialtion fans");
    if(tempWarn == 0){
      Blynk.logEvent("temperature_",String("High temperature detected in cabin . Activation the auto cooling system to prevent the dashboard components from getting damaged. Please check your car as soon as you can . Current Temperature is :- ") + t);
      tempWarn = 1;
    }
  }else if(t>70 && t<75){
    analogWrite(enable2Pin,200);
    Blynk.virtualWrite(V3,"Warning! High temperature");
    Blynk.virtualWrite(V4,"Activating ventialtion fans");
    if(tempWarn == 0){
      Blynk.logEvent("temperature_",String("High temperature detected in cabin . Activation the auto cooling system to prevent the dashboard components from getting damaged. Please check your car as soon as you can . Current Temperature is :- ") + t);
      tempWarn = 1;
    }
  }else if(t>75){
    analogWrite(enable2Pin,255);
    Blynk.virtualWrite(V3,"Warning! High temperature");
    Blynk.virtualWrite(V4,"Activating ventialtion fans");
    if(tempWarn == 0){
      Blynk.logEvent("temperature_",String("High temperature detected in cabin . Activation the auto cooling system to prevent the dashboard components from getting damaged. Please check your car as soon as you can . Current Temperature is :- ") + t);
      tempWarn = 1;
    }
  }else{
    analogWrite(enable2Pin,0);
    tempWarn = 0;
  }
  if(Serial.available()){
      data = Serial.readString();
      data.trim();
      Serial.println(data);
      if(data == "Unauthorised"){
        faceNotMatching = 1;
        Blynk.logEvent("unauthorised_");
        Blynk.virtualWrite(V3,"Unauthorised Person");
        Blynk.virtualWrite(V4,"Im slowing down");
        digitalWrite(LED_BUILTIN, HIGH);
        // speed=21;
        // analogWrite(motor,speed);
      }if(data == "NP"){     
        digitalWrite(LED_BUILTIN, HIGH);     
        engineOn = 0;
        speed = 0;
        analogWrite(motor,0);
        Blynk.virtualWrite(V1,0);
        Blynk.virtualWrite(V2,0);
        Blynk.virtualWrite(V7,0);
        Blynk.virtualWrite(V3,"No person");
        Blynk.virtualWrite(V4,"Im going to sleep!");
        digitalWrite(12,LOW);
        Blynk.virtualWrite(V6,0);
      }if(data == "ID"){
        IncorrectID = 1;
        digitalWrite(LED_BUILTIN, HIGH);
        Blynk.virtualWrite(V3,"ID Not Matching");
        Blynk.virtualWrite(V4,"Im slowing down!");
        // speed=21;
        // analogWrite(motor,speed);          
      }if(data == "DR"){
        drowsy = 1;
        digitalWrite(LED_BUILTIN, HIGH);
        Blynk.virtualWrite(V3,"Drowsiness detected");
        Blynk.logEvent("drowsiness_");
        Blynk.virtualWrite(V4,"Im slowing down!");
      }if(data == "NDR"){
        drowsy = 0;
        Blynk.virtualWrite(V3,"Happy Ride!");
        Blynk.virtualWrite(V4,"Drive safe, its not GTA.");
        digitalWrite(LED_BUILTIN, LOW);
      }if(data == "OK"){
        NoMatching = 0;
        faceNotMatching = 0;
        Blynk.virtualWrite(V3,"Happy Ride!");
        Blynk.virtualWrite(V4,"Drive safe, its not GTA.");
        IncorrectID = 0;
        digitalWrite(LED_BUILTIN, LOW);
      }if(data == "DONE"){
        Blynk.virtualWrite(V6,0);
      }
      delay(10);
    }
}
