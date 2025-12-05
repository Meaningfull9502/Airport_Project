import os
import pandas as pd
from pathlib import Path

# logs 폴더가 있는 최상위 경로 (스크립트 위치 기준)
# 만약 스크립트가 logs 폴더 상위에 있다면 './logs'
# 스크립트가 logs 폴더 안에 있다면 '.' 로 설정하세요.

def merge_csv_files(root_dir, n):
    all_dataframes = []
    
    # pathlib을 사용하여 하위 폴더까지 재귀적으로(recursive) 탐색
    # rglob('*')는 모든 파일을 찾지만, 우리는 특정 파일명만 필요하므로 필터링
    target_filename = f'{n}_models_comparison.csv'
    
    print(f"'{root_dir}' 경로에서 '{target_filename}' 파일을 찾는 중...")
    
    files_found = list(Path(root_dir).rglob(target_filename))
    
    if not files_found:
        print("파일을 찾지 못했습니다. 경로를 확인해주세요.")
        return

    for file_path in files_found:
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            
            # 중요: 어느 폴더의 데이터인지 구분하기 위해 폴더명 컬럼 추가
            # file_path.parent.name 은 'arrival_56_False' 같은 폴더명을 가져옴
            df['source_folder'] = file_path.parent.name
            
            # 리스트에 추가
            all_dataframes.append(df)
            print(f"읽기 성공: {file_path}")
            
        except Exception as e:
            print(f"에러 발생 ({file_path}): {e}")

    if all_dataframes:
        # 모든 데이터프레임 합치기
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # 'source_folder' 컬럼을 맨 앞으로 이동 (보기 편하게)
        cols = ['source_folder'] + [col for col in merged_df.columns if col != 'source_folder']
        merged_df = merged_df[cols]
        
        # 결과 저장
        merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig') # 한글 깨짐 방지 utf-8-sig
        print(f"\n성공적으로 합쳐졌습니다! -> {output_filename}")
        print(f"총 {len(merged_df)} 개의 행이 저장되었습니다.")
    else:
        print("합칠 데이터가 없습니다.")


if __name__ == "__main__":
    root_path = './logs' 
    for n in ['arrival', 'departure', 'both']:
        output_filename = f'merged_{n}_models_comparison.csv'
        merge_csv_files(root_path, n)