#include "DHT.h"

const int infraredPin = 3;
const int mq2Pin = A0;
#define DHTPIN 2
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

const int redLedPin = 9;
const int yellowLedPin = 10;
const int greenLedPin = 11;
const int buzzerPin = 8;
const int waterPumpPin = 7;

const int MQ2_THRESHOLD_FIRE = 400;
const float TEMP_THRESHOLD_FIRE = 35.0;
const float HUMIDITY_THRESHOLD_FIRE = 50.0;

const int MQ2_THRESHOLD_WARNING = 250;
const float TEMP_THRESHOLD_WARNING = 30.0;

void setup() {
  Serial.begin(9600);

  pinMode(infraredPin, INPUT);

  pinMode(redLedPin, OUTPUT);
  pinMode(yellowLedPin, OUTPUT);
  pinMode(greenLedPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(waterPumpPin, OUTPUT);

  digitalWrite(redLedPin, LOW);
  digitalWrite(yellowLedPin, LOW);
  digitalWrite(greenLedPin, LOW);
  digitalWrite(buzzerPin, LOW);
  digitalWrite(waterPumpPin, LOW);

  dht.begin();
  Serial.println("Arduino Uno Fire Detector Ready!");
  Serial.println("Mode: Standalone AI (Berbasis Aturan)");
  Serial.println("------------------------------------");
}

void loop() {
  int infraredValue = digitalRead(infraredPin);
  int mq2Value = analogRead(mq2Pin);
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature)) {
    humidity = -1.0;
    temperature = -1.0;
    Serial.println("DHT11 read failed!");
  }

  String currentStatus = "NORMAL";

  bool isInfraredDetectingFire = (infraredValue == LOW);

  bool isGasHigh = (mq2Value > MQ2_THRESHOLD_FIRE);
  bool isTempHigh = (temperature > TEMP_THRESHOLD_FIRE && temperature != -1.0);
  bool isHumidityLow = (humidity < HUMIDITY_THRESHOLD_FIRE && humidity != -1.0);

  if (isInfraredDetectingFire || (isGasHigh && isTempHigh && isHumidityLow)) {
    currentStatus = "FIRE";
  }
  else if ((mq2Value > MQ2_THRESHOLD_WARNING) || (temperature > TEMP_THRESHOLD_WARNING && temperature != -1.0)) {
    currentStatus = "WARNING";
  }
  else {
    currentStatus = "NORMAL";
  }

  if (currentStatus == "NORMAL") {
    digitalWrite(greenLedPin, HIGH);
    digitalWrite(yellowLedPin, LOW);
    digitalWrite(redLedPin, LOW);
    digitalWrite(buzzerPin, LOW);
    digitalWrite(waterPumpPin, LOW);
  } else if (currentStatus == "WARNING") {
    digitalWrite(greenLedPin, LOW);
    digitalWrite(yellowLedPin, HIGH);
    digitalWrite(redLedPin, LOW);
    digitalWrite(buzzerPin, LOW);
    digitalWrite(waterPumpPin, LOW);
  } else if (currentStatus == "FIRE") {
    digitalWrite(greenLedPin, LOW);
    digitalWrite(yellowLedPin, LOW);
    digitalWrite(redLedPin, HIGH);
    digitalWrite(buzzerPin, HIGH);
    digitalWrite(waterPumpPin, HIGH);
  }

  Serial.print(infraredValue);
  Serial.print(",");
  Serial.print(mq2Value);
  Serial.print(",");
  Serial.print(humidity, 1);
  Serial.print(",");
  Serial.print(temperature, 1);
  Serial.print(",");
  Serial.println(currentStatus);

  delay(500);
}
