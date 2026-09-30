"""
DataVolley(.dvw) 세터 의사결정 분석 - 배치 전처리 파이프라인
================================================================
사용법 (Jupyter Notebook):

    import sys
    sys.path.insert(0, '.')  # 이 파일이 있는 경로
    from dvw_batch_pipeline import run_batch

    TARGET_SETTER_IDS = ['TEAM_X-0001', 'TEAM_X-0002', 'TEAM_Y-0001', ...]  # 9인 세터 player_id (예시)
    final_df, error_log = run_batch(
        folder_path='./dvw_files',   # 126개 .dvw 파일이 들어있는 폴더
        target_setter_ids=TARGET_SETTER_IDS,
    )
    final_df.to_csv('final_dataset.csv', index=False, encoding='utf-8-sig')

인코딩: 한국어 DataVolley 파일은 기본적으로 cp949로 읽습니다.
        (다른 인코딩이 섞여 있으면 load_dvw(..., encoding=...) 로 개별 조정)

주의: 10번(팀 내 에이스 전/후위), 15번(공격수별 시즌 성공률)은
      외부 시즌 통계 파일이 있어야 하는 항목이라 이 스크립트에는 없습니다.
      시즌 통계 파일(선수 player_id 기준)을 주시면 이 스크립트가 만든
      final_df 에 player_id로 merge하는 코드만 추가하면 됩니다.
"""

import os
import glob
import re
import uuid
import traceback

import pandas as pd
import numpy as np


# ================================================================
# 0. dvw 파일 저수준 로더 (섹션 파싱)
# ================================================================


def extract_num_blockers(code):
    """
    공격 코드에서 블로커 수 추출.
    형식 예: 'a23AQ#XC~36CH1' -> 좌표(~36) 이후의 H/P/T + 숫자 = 블로킹 인원 수.
    주의 1: attack_code 자체가 'P8'/'P9'(파이프 공격)일 수 있어, 반드시 좌표 구분자(~) 이후만 탐색해야
            공격조합 코드의 숫자를 블로커 수로 오인하지 않음.
    주의 2: 숫자 '4'는 "블로커 4명"이 아니라 "블로커 사이(seam)로 공격이 들어감"을 뜻하는
            별도 코드임 (실제 인원수가 아님) -> 이 함수는 원본 숫자를 그대로 반환하고,
            숫자→의미 변환은 아래 classify_block_situation()에서 별도 처리.
    """
    if pd.isna(code) or len(code) < 10:
        return np.nan
    tail = code[9:]
    m = re.search(r'[HPT](\d)', tail)
    return int(m.group(1)) if m else np.nan


def classify_block_situation(raw_value):
    """
    extract_num_blockers()의 원본 숫자를 해석:
      0,1,2,3 -> 실제 블로커 인원수 (그대로 사용)
      4       -> '블로커 사이(seam)'로 공격 -> 실제 인원수가 아니므로 별도 카테고리
    반환: (num_blockers_actual: 0~3 또는 NaN, is_between_blockers: bool)
    """
    if pd.isna(raw_value):
        return np.nan, np.nan
    if raw_value == 4:
        return np.nan, True  # 실제 인원수는 알 수 없음(혹은 정의상 해당없음), 다만 '사이 공격' 플래그는 True
    return raw_value, False


def _decode_backup_field(s):
    """
    '\x0f' 로 시작하는 유니코드 백업 필드(문자열, 이미 cp949로 디코딩된 상태) 복호화.
    형식: \x0f + 인코딩타입숫자(2=UTF-8, 4=UTF-16BE) + 헥스문자열
    """
    if not isinstance(s, str) or not s.startswith('\x0f') or len(s) < 2:
        return None
    try:
        type_digit = s[1]
        hexstr = s[2:]
        if not hexstr:
            return ''
        data = bytes.fromhex(hexstr)
        if type_digit == '2':
            return data.decode('utf-8', errors='replace')
        elif type_digit == '4':
            return data.decode('utf-16-be', errors='replace')
    except (ValueError, IndexError):
        return None
    return None


def _looks_corrupted(s):
    """이름/팀명이 '?'로만 이루어져 있으면(원본 파일 자체 인코딩 손상) True."""
    return isinstance(s, str) and len(s) > 0 and re.fullmatch(r'\?+', s) is not None


def detect_dv_encoding(file_path, candidates=('cp949', 'utf-8', 'cp1252', 'latin-1')):
    """
    dvw 파일에 어떤 인코딩이 맞는지 확인하는 헬퍼.
    에러 없이 디코딩됐다고 해서 한글이 정확한 건 아니므로(모지바케 가능),
    실제로 이름 몇 개를 출력해서 눈으로 확인하는 걸 권장.
    """
    with open(file_path, 'rb') as f:
        raw = f.read()
    results = {}
    for enc in candidates:
        try:
            raw.decode(enc)
            results[enc] = 'OK (에러 없이 디코딩됨)'
        except UnicodeDecodeError as e:
            results[enc] = f'실패: {e}'
    return results


def parse_settercall_table(file_path, encoding='cp949'):
    """[3SETTERCALL] 섹션 -> {코드: 설명}"""
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        lines = f.readlines()
    lines = [l.rstrip('\r\n') for l in lines]
    if '[3SETTERCALL]' not in lines:
        return {}
    start = lines.index('[3SETTERCALL]') + 1
    table = {}
    for l in lines[start:]:
        if l.startswith('['):
            break
        parts = l.split(';')
        if len(parts) >= 3 and parts[0]:
            table[parts[0]] = parts[2]
    return table


def parse_attackcombination_table(file_path, encoding='cp949'):
    """[3ATTACKCOMBINATION] 섹션 -> {공격코드: {'zone':.., 'desc':..}}"""
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        lines = f.readlines()
    lines = [l.rstrip('\r\n') for l in lines]
    if '[3ATTACKCOMBINATION]' not in lines:
        return {}
    start = lines.index('[3ATTACKCOMBINATION]') + 1
    table = {}
    for l in lines[start:]:
        if l.startswith('['):
            break
        parts = l.split(';')
        if len(parts) >= 5 and parts[0]:
            table[parts[0]] = {'zone': parts[1], 'desc': parts[4]}
    return table


def classify_settercall(desc):
    """세터콜 설명 -> 'Middle' / 'Shift' / 'Other'"""
    if pd.isna(desc):
        return np.nan
    d = str(desc).lower()
    if any(k in d for k in ['quick', 'mb', 'center']):
        return 'Middle'
    if 'shifted to 2' in d or 'shifted to 4' in d:
        return 'Shift'
    return 'Other'


def load_players(meta_data, team_name, H_or_V):
    cols_rename = {
        1: "player_number", 3: "starting_position_set1", 4: "starting_position_set2",
        5: "starting_position_set3", 6: "starting_position_set4", 7: "starting_position_set5",
        8: "player_id", 9: "lastname", 10: "firstname", 11: "nickname",
        12: "special_role", 13: "role", 14: "foreign"
    }
    team_players = meta_data[
        (meta_data['meta_group'] == f'3PLAYERS-{H_or_V}') &
        (~meta_data[0].str.strip().eq(f'[3PLAYERS-{H_or_V}]'))
    ][0].str.split(';', expand=True)

    # 유니코드 백업 필드(있으면): lastname->17, firstname->18, nickname->19 (원본 컬럼 기준)
    backup_cols = {9: 17, 10: 18, 11: 19}
    for primary_idx, backup_idx in backup_cols.items():
        if primary_idx not in team_players.columns or backup_idx not in team_players.columns:
            continue
        primary = team_players[primary_idx]
        backup_decoded = team_players[backup_idx].map(_decode_backup_field)
        needs_fix = primary.map(_looks_corrupted) & backup_decoded.notna() & (backup_decoded != '')
        team_players[primary_idx] = np.where(needs_fix, backup_decoded, primary)

    team_players.columns = [cols_rename.get(c, c) for c in team_players.columns]
    for c in ['nickname', 'firstname', 'lastname']:
        team_players[c] = team_players[c].fillna('').str.strip()
    team_players['player_name'] = (team_players['firstname'] + ' ' + team_players['lastname']).str.strip()
    team_players['team'] = team_name
    keep_cols = ['player_number', 'player_id', 'player_name', 'team',
                 'starting_position_set1', 'starting_position_set2', 'starting_position_set3',
                 'starting_position_set4', 'starting_position_set5']
    return team_players[keep_cols]


def load_dvw(file_path, encoding='cp949'):
    """
    dvw 파일 1개 -> (plays DataFrame, players DataFrame)
    한국어 파일은 cp949가 기본. 다른 인코딩이면 encoding= 인자로 지정.
    """
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        rows = f.readlines()

    full_file = pd.DataFrame(rows)
    full_file['meta_group'] = full_file[0].str.extract(r'\[(.*?)\]', expand=False).ffill()

    def _section_index(tag):
        return full_file.index[full_file[0].str.strip() == tag][0]

    t_idx = _section_index('[3TEAMS]')
    home_row = rows[t_idx + 1].strip().split(';')
    visit_row = rows[t_idx + 2].strip().split(';')
    home_team_id, home_team = home_row[0], home_row[1]
    visiting_team_id, visiting_team = visit_row[0], visit_row[1]

    # 팀명 원본이 손상(전부 '?')됐으면 유니코드 백업 필드(인덱스 6)로 복구
    if _looks_corrupted(home_team) and len(home_row) > 6:
        backup = _decode_backup_field(home_row[6])
        if backup:
            home_team = backup
    if _looks_corrupted(visiting_team) and len(visit_row) > 6:
        backup = _decode_backup_field(visit_row[6])
        if backup:
            visiting_team = backup

    meta_data = full_file[full_file['meta_group'] != '3SCOUT']
    players = pd.concat([
        load_players(meta_data, home_team, 'H'),
        load_players(meta_data, visiting_team, 'V')
    ], ignore_index=True)

    scout_idx = _section_index('[3SCOUT]')
    plays = full_file.iloc[scout_idx + 1:].reset_index(drop=True)
    plays = plays[0].str.split(';', expand=True)

    rename_map = {0: 'code', 1: 'point_phase_raw', 2: 'attack_phase_raw',
                  4: 'start_coordinate', 5: 'mid_coordinate', 6: 'end_coordinate',
                  7: 'time', 8: 'set_number', 9: 'home_setter_position',
                  10: 'visiting_setter_position', 11: 'video_file_number', 12: 'video_time'}
    plays = plays.rename(columns=rename_map)
    new_cols = list(plays.columns)
    for i in range(6):
        new_cols[14 + i] = f'home_p{i+1}'
        new_cols[20 + i] = f'visiting_p{i+1}'
    plays.columns = new_cols
    plays = plays.drop(columns=[3, 13, 26])

    plays['match_id'] = str(uuid.uuid4())

    coord_map = {'-1-1': np.nan}
    for c in ['start_coordinate', 'mid_coordinate', 'end_coordinate']:
        plays[c] = plays[c].map(lambda v: coord_map.get(v, v))

    plays['team'] = np.where(plays['code'].str[0:1] == '*', home_team, visiting_team)
    plays['player_number'] = plays['code'].str[1:3].str.extract(r'(\d{2})').astype(float).fillna(0).astype(int).astype(str)
    plays['player_number'] = np.where(plays['player_number'] == '0', np.nan, plays['player_number'])

    plays = pd.merge(plays, players, on=['player_number', 'team'], how='left')

    skill_map = {"S": "Serve", "R": "Reception", "E": "Set", "A": "Attack",
                 "D": "Dig", "B": "Block", "F": "Freeball", "p": "Point"}

    def _skill(row):
        if pd.isna(row['player_number']):
            return np.nan
        return row['code'][3] if len(row['code']) > 3 else np.nan

    plays['skill'] = plays.apply(_skill, axis=1).map(skill_map)
    plays['skill'] = np.where(plays['code'].str[1:2] == 'p', 'Point', plays['skill'])

    eval_codes = ["#", "+", "!", "-", "/", "="]
    plays['evaluation_code'] = plays['code'].str.slice(5, 6)
    plays['evaluation_code'] = np.where(plays['evaluation_code'].isin(eval_codes), plays['evaluation_code'], np.nan)

    plays['set_code'] = np.where(plays['skill'] == 'Set', plays['code'].str.slice(6, 8), np.nan)
    plays['set_code'] = np.where((plays['skill'] == 'Set') & (plays['set_code'] != '~~'), plays['set_code'], np.nan)

    plays['attack_code'] = plays['code'].str.slice(6, 8)
    plays['attack_code'] = np.where((plays['skill'] == 'Attack') & (plays['attack_code'] != '~~'), plays['attack_code'], np.nan)

    plays['start_zone'] = plays['code'].str.slice(9, 10).replace({'~': np.nan, '': np.nan})
    plays['end_zone'] = plays['code'].str.slice(10, 11).replace({'~': np.nan, '': np.nan})
    plays['end_subzone'] = plays['code'].str.slice(11, 12).replace({'~': np.nan, '': np.nan})

    plays['rally_number'] = plays.groupby('set_number', group_keys=False)['skill'].apply(lambda x: (x == 'Serve').cumsum())

    plays['possession_number'] = plays.groupby(
        ['set_number', 'rally_number'], group_keys=False
    )['skill'].apply(lambda x: (x == 'Attack').shift(1).cumsum() + 1).fillna(0).astype(int)

    plays['point_won_by'] = np.select(
        [plays['code'].str[0:2] == '*p', plays['code'].str[0:2] == 'ap'],
        [home_team, visiting_team], default=None
    )
    plays['point_won_by'] = plays['point_won_by'].bfill()

    plays['home_team_score'] = plays[plays['code'].str[1:2] == 'p']['code'].str.slice(2, 4)
    plays['home_team_score'] = plays.groupby(['set_number', 'rally_number'])['home_team_score'].bfill()
    plays['home_team_score'] = pd.to_numeric(plays['home_team_score'], errors='coerce').astype('Int64')

    plays['visiting_team_score'] = plays[plays['code'].str[1:2] == 'p']['code'].str.slice(5, 7)
    plays['visiting_team_score'] = plays.groupby(['set_number', 'rally_number'])['visiting_team_score'].bfill()
    plays['visiting_team_score'] = pd.to_numeric(plays['visiting_team_score'], errors='coerce').astype('Int64')

    plays['serving_team'] = np.where((plays['skill'] == 'Serve') & (plays['code'].str[0:1] == '*'), home_team, None)
    plays['serving_team'] = np.where((plays['skill'] == 'Serve') & (plays['code'].str[0:1] == 'a'), visiting_team, plays['serving_team'])
    plays['serving_team'] = plays.groupby(['set_number', 'rally_number'])['serving_team'].ffill()
    plays['receiving_team'] = np.where(plays['serving_team'] == home_team, visiting_team, home_team)
    plays['receiving_team'] = np.where(plays['serving_team'].isna(), np.nan, plays['receiving_team'])

    plays['home_team'] = home_team
    plays['visiting_team'] = visiting_team
    plays['home_team_id'] = home_team_id
    plays['visiting_team_id'] = visiting_team_id

    return plays, players


# ================================================================
# 1~2. 점수 정제 + 실시간 스트릭
# ================================================================

def _reset_set_scores(group):
    first_change_idx = (group['h_score'].diff().fillna(0) != 0) | (group['v_score'].diff().fillna(0) != 0)
    if first_change_idx.any():
        first_point_loc = group.index[first_change_idx][0]
        group.loc[:first_point_loc - 1, ['h_score', 'v_score']] = 0
    else:
        group[['h_score', 'v_score']] = 0
    return group


def _get_realtime_streaks(df):
    h_streaks, v_streaks = [], []
    h_cur, v_cur = 0, 0
    last_h, last_v = 0, 0
    for _, row in df.iterrows():
        curr_h, curr_v = row['h_score'], row['v_score']
        if curr_h > last_h:
            h_cur = h_cur + 1 if h_cur > 0 else 1
            v_cur = v_cur - 1 if v_cur < 0 else -1
        elif curr_v > last_v:
            v_cur = v_cur + 1 if v_cur > 0 else 1
            h_cur = h_cur - 1 if h_cur < 0 else -1
        h_streaks.append(h_cur); v_streaks.append(v_cur)
        last_h, last_v = curr_h, curr_v
    df['h_streak'] = h_streaks; df['v_streak'] = v_streaks
    return df


def _run_length(sub):
    """같은 값이 연속으로 몇 번째인지 (1부터 시작)"""
    change = sub != sub.shift(1)
    grp = change.cumsum()
    return sub.groupby(grp).cumcount() + 1


# ================================================================
# 3. 파일 1개 처리 -> wide format final_data
# ================================================================

RECEIVE_GRADE_MAP = {'#': 3, '+': 2, '!': 1, '-': 0}   # 항목0
RESULT_MAP = {'#': '1', '=': '2', '/': '2', '+': '3', '!': '3', '-': '3'}  # 항목9
PRIOR_SKILLS = ['Reception', 'Dig', 'Freeball']


def load_season_stats(csv_path):
    """
    시즌 공격 통계 CSV 로드 + 팀별 에이스(Pts 최다 1명) 산출.
    반환: (season_df, ace_df[team, player_number])
    """
    season = pd.read_csv(csv_path)
    season['player_number'] = pd.to_numeric(season['player_number'], errors='coerce').astype('Int64').astype(str)
    season['Pts'] = pd.to_numeric(season['Pts'], errors='coerce')
    season['season_attack_pct'] = pd.to_numeric(season['season_attack_pct'], errors='coerce')
    ace_df = season.loc[season.groupby('team')['Pts'].idxmax(), ['team', 'player_number']].rename(
        columns={'player_number': 'ace_number'}
    )
    return season, ace_df


def resolve_target_setter_ids(players, target_setter_map):
    """
    target_setter_map: {'TEAM_C': [2, 3], 'TEAM_G': [2], ...} 형태의 팀별 등번호 딕셔너리
    -> 이 파일(경기)의 실제 player_id 리스트로 변환.
       해당 팀/번호 선수가 이 경기 로스터에 없으면 그냥 빠짐 (에러 아님).
    """
    ids = []
    for team, numbers in target_setter_map.items():
        numbers_str = [str(n) for n in numbers]
        match = players[(players['team'] == team) & (players['player_number'].isin(numbers_str))]
        ids.extend(match['player_id'].tolist())
    return ids


def process_single_file(file_path, target_setters, encoding='cp949', season_df=None, ace_df=None):
    """
    dvw 파일 1개 -> 세터 시퀀스(R/D/F -> Set -> Attack) 기반 wide-format final_data

    target_setters: 아래 둘 중 하나
      - dict: {'TEAM_C': [2, 3], 'TEAM_G': [2], ...}  (팀명 + 등번호, 권장)
      - list: ['TEAM_X-0001', 'TEAM_Y-0001', ...]         (player_id 직접 지정, 이전 방식)

    season_df, ace_df: load_season_stats()의 반환값. None이면 10/15번 컬럼 없이 진행.
    """
    data2, players = load_dvw(file_path, encoding=encoding)
    data_clean = data2.dropna(subset=['skill']).reset_index(drop=True)

    if isinstance(target_setters, dict):
        target_setter_ids = resolve_target_setter_ids(players, target_setters)
    else:
        target_setter_ids = list(target_setters)

    # --- 항목15: 공격수별 시즌 공격성공률 매핑 (team+player_number 기준) ---
    if season_df is not None:
        players_with_season = players.merge(
            season_df[['team', 'player_number', 'season_attack_pct']],
            on=['team', 'player_number'], how='left'
        )
        season_pct_map = players_with_season.set_index('player_id')['season_attack_pct'].to_dict()
        data_clean['season_attack_pct'] = data_clean['player_id'].map(season_pct_map)
    else:
        data_clean['season_attack_pct'] = np.nan

    # --- 항목10: 팀 내 에이스(시즌 Pts 최다)의 전위/후위 여부 ---
    #   세터처럼 실시간으로 기록되는 필드가 아니라서, 세트 시작 포지션 + 세터 로테이션 오프셋으로 역산.
    #   현재 세터 포지션(home/visiting_setter_position, 랠리마다 기록됨)과
    #   그 세터의 "세트 시작 포지션"의 차이(오프셋)를 에이스의 시작 포지션에 그대로 더해서
    #   에이스의 현재 포지션을 추정 -> 2/3/4면 전위.
    data_clean['is_team_ace'] = False
    data_clean['ace_current_position'] = np.nan
    if ace_df is not None:
        players_with_ace = players.merge(ace_df, on='team', how='left')
        players_with_ace['is_ace'] = players_with_ace['player_number'] == players_with_ace['ace_number']
        is_ace_map = players_with_ace.set_index('player_id')['is_ace'].to_dict()
        data_clean['is_team_ace'] = data_clean['player_id'].map(is_ace_map).fillna(False)

        pos_cols = {1: 'starting_position_set1', 2: 'starting_position_set2', 3: 'starting_position_set3',
                    4: 'starting_position_set4', 5: 'starting_position_set5'}
        ace_rows = players_with_ace[players_with_ace['is_ace']].set_index('team')

        for team_name in [data_clean['home_team'].iloc[0], data_clean['visiting_team'].iloc[0]]:
            if team_name not in ace_rows.index:
                continue
            ace_row = ace_rows.loc[team_name]
            is_home = (team_name == data_clean['home_team'].iloc[0])
            setter_pos_col = 'home_setter_position' if is_home else 'visiting_setter_position'

            for set_num in data_clean['set_number'].dropna().unique():
                mask = (data_clean['set_number'] == set_num) & (data_clean['team'] == team_name)
                if not mask.any():
                    continue
                ace_start = pos_cols.get(int(set_num))
                ace_start_pos = ace_row.get(ace_start) if ace_start else None
                if ace_start_pos in (None, '', '*') or pd.isna(ace_start_pos):
                    continue
                ace_start_pos = int(ace_start_pos)

                team_set_mask = (data_clean['set_number'] == set_num)
                setter_pos_series = pd.to_numeric(data_clean.loc[team_set_mask, setter_pos_col], errors='coerce')
                if setter_pos_series.dropna().empty:
                    continue
                setter_start_pos = setter_pos_series.dropna().iloc[0]

                current_setter_pos = pd.to_numeric(data_clean.loc[mask, setter_pos_col], errors='coerce')
                offset = current_setter_pos - setter_start_pos
                ace_current = ((ace_start_pos - 1 + offset) % 6) + 1
                data_clean.loc[mask, 'ace_current_position'] = ace_current

    data_clean['ace_is_front'] = data_clean['ace_current_position'].isin([2, 3, 4]).astype('Int64')
    data_clean.loc[data_clean['ace_current_position'].isna(), 'ace_is_front'] = pd.NA

    if data_clean.empty:
        raise ValueError('스킬 데이터가 비어있음 (파싱 실패 가능성)')

    # --- 점수 정제 ---
    data_clean['h_score'] = pd.to_numeric(data_clean['home_team_score'], errors='coerce').fillna(0).astype(int)
    data_clean['v_score'] = pd.to_numeric(data_clean['visiting_team_score'], errors='coerce').fillna(0).astype(int)
    fixed_scores = data_clean.groupby(['match_id', 'set_number'], group_keys=False)[['h_score', 'v_score']].apply(_reset_set_scores)
    data_clean[['h_score', 'v_score']] = fixed_scores

    # --- 실시간 스트릭 ---
    fixed_streaks = data_clean.groupby(['match_id', 'set_number'], group_keys=False)[['h_score', 'v_score']].apply(_get_realtime_streaks)
    data_clean[['h_streak', 'v_streak']] = fixed_streaks[['h_streak', 'v_streak']]
    data_clean['actual_streak_flow'] = np.where(
        data_clean['team'] == data_clean['home_team'], data_clean['h_streak'], data_clean['v_streak']
    )

    # --- 항목0: 리시브/디그/프리볼 정확도 (4단계) ---
    data_clean['touch_quality_score'] = np.where(
        data_clean['skill'].isin(PRIOR_SKILLS),
        data_clean['evaluation_code'].map(RECEIVE_GRADE_MAP),
        np.nan
    )

    # --- 항목3: 선수별 실시간(직전까지 누적) 공격 성공률 (성공기준: '#') ---
    attack_mask = data_clean['skill'] == 'Attack'
    attacks = data_clean[attack_mask].copy()
    attacks['is_kill'] = (attacks['evaluation_code'] == '#').astype(int)
    attacks['is_attempt'] = attacks['evaluation_code'].notna().astype(int)
    attacks['cum_kills_prior'] = attacks.groupby('player_id')['is_kill'].transform(lambda s: s.cumsum().shift(1).fillna(0))
    attacks['cum_attempts_prior'] = attacks.groupby('player_id')['is_attempt'].transform(lambda s: s.cumsum().shift(1).fillna(0))
    attacks['rt_atk_eff'] = np.where(attacks['cum_attempts_prior'] > 0,
                                      attacks['cum_kills_prior'] / attacks['cum_attempts_prior'], np.nan)
    data_clean.loc[attacks.index, 'rt_atk_eff'] = attacks['rt_atk_eff']

    # --- 항목16: 세터콜 설명/분류, 항목4용 공격존 ---
    settercall_table = parse_settercall_table(file_path, encoding=encoding)
    attackcombo_table = parse_attackcombination_table(file_path, encoding=encoding)
    data_clean['settercall_desc'] = data_clean['set_code'].map(settercall_table)
    data_clean['settercall_type'] = data_clean['settercall_desc'].map(classify_settercall)
    data_clean['attack_zone'] = data_clean['attack_code'].map(
        lambda c: attackcombo_table.get(c, {}).get('zone') if pd.notna(c) else np.nan
    )

    # 블로커 수 (Attack 행에만 의미 있음)
    # raw: 0~4 원본 숫자 (4='블로커 사이' 특수 코드, 실제 인원수 아님)
    raw_block = np.where(
        data_clean['skill'] == 'Attack',
        data_clean['code'].map(extract_num_blockers),
        np.nan
    )
    parsed = [classify_block_situation(v) for v in raw_block]
    data_clean['num_blockers'] = [p[0] for p in parsed]           # 0~3 실제 인원수 (4는 NaN 처리)
    data_clean['attack_between_blockers'] = [p[1] for p in parsed]  # 4('사이') 여부 플래그

    # --- 세터 필터 (외부에서 지정한 player_id 리스트) ---
    setter_filter = data_clean['player_id'].isin(target_setter_ids)

    # --- R/D/F -> Set -> Attack 시퀀스 추출 ---
    set_indices = data_clean.index[(data_clean['skill'] == 'Set') & setter_filter].tolist()
    records = []
    seq_id = 0
    for idx in set_indices:
        prev_idx, next_idx = idx - 1, idx + 1
        if prev_idx < 0 or next_idx >= len(data_clean):
            continue
        prev_row, curr_row, next_row = data_clean.iloc[prev_idx], data_clean.iloc[idx], data_clean.iloc[next_idx]
        same_rally = (prev_row['rally_number'] == curr_row['rally_number'] == next_row['rally_number'])
        same_team = (prev_row['team'] == curr_row['team'] == next_row['team'])
        pattern_ok = (prev_row['skill'] in PRIOR_SKILLS) and (next_row['skill'] == 'Attack')
        if same_rally and same_team and pattern_ok:
            seq_id += 1
            for role, row in zip(['prior_touch', 'set', 'attack'], [prev_row, curr_row, next_row]):
                rec = row.copy()
                rec['sequence_id'] = seq_id
                rec['sequence_role'] = role
                records.append(rec)

    if not records:
        return pd.DataFrame()  # 이 파일엔 대상 세터의 유효 시퀀스가 없음

    data3 = pd.DataFrame(records).reset_index(drop=True)

    first_touch_df = data3.iloc[0::3].reset_index(drop=True)
    set_df = data3.iloc[1::3].reset_index(drop=True)
    attack_df = data3.iloc[2::3].reset_index(drop=True)

    # 항목9: 팀 기준 직전 공격 결과
    current_results = attack_df['evaluation_code'].map(RESULT_MAP)
    prev_attack_results = current_results.groupby(set_df['team']).shift(1).fillna('0')

    temp_own_score, temp_opp_score, own_rot, opp_rot, game_stages = [], [], [], [], []
    for _, row in set_df.iterrows():
        h_s = int(np.nan_to_num(pd.to_numeric(row['home_team_score'], errors='coerce')))
        v_s = int(np.nan_to_num(pd.to_numeric(row['visiting_team_score'], errors='coerce')))
        max_s = max(h_s, v_s)
        stage = 1 if max_s < 8 else 2 if max_s < 16 else 3 if max_s < 20 else 4  # 항목2
        game_stages.append(stage)
        if row['team'] == row['home_team']:
            temp_own_score.append(h_s); temp_opp_score.append(v_s)
            own_rot.append(row['home_setter_position']); opp_rot.append(row['visiting_setter_position'])
        else:
            temp_own_score.append(v_s); temp_opp_score.append(h_s)
            own_rot.append(row['visiting_setter_position']); opp_rot.append(row['home_setter_position'])

    score_diff_arr = np.array(temp_own_score) - np.array(temp_opp_score)  # 항목1

    # 항목4: 직전 공격 "존" 반복 여부 (같은 세터 기준)
    setter_id_series = set_df['player_id'].reset_index(drop=True)
    attack_zone_series = attack_df['attack_zone'].reset_index(drop=True)
    prev_attack_zone = attack_zone_series.groupby(setter_id_series).shift(1)
    is_repeat_route = (attack_zone_series == prev_attack_zone).astype('Int64')

    # 항목11: 같은 세터 -> 같은 공격수 연속 토스 횟수
    attacker_id_series = attack_df['player_id'].reset_index(drop=True)
    consecutive_toss_count = attacker_id_series.groupby(setter_id_series).transform(_run_length)

    # 항목12: 사이드아웃/브레이크포인트 (서브권 팀 기준)
    point_type = np.where(set_df['team'].values == set_df['serving_team'].values, 'Breakpoint', 'Sideout')

    # 항목13: 클러치 상황 (20점 이후 + 2점차 이내)
    is_clutch = ((np.array(game_stages) == 4) & (np.abs(score_diff_arr) <= 2)).astype(int)

    final_data = pd.DataFrame({
        'match_id': set_df['match_id'].reset_index(drop=True),
        'source_file': os.path.basename(file_path),
        'set_number': set_df['set_number'].reset_index(drop=True),
        'team': set_df['team'].reset_index(drop=True),

        'own_score': temp_own_score, 'opp_score': temp_opp_score,           # 항목6
        'h_score': set_df['h_score'].reset_index(drop=True),
        'v_score': set_df['v_score'].reset_index(drop=True),

        'first_touch': first_touch_df['skill'].reset_index(drop=True),
        'touch_quality': first_touch_df['evaluation_code'].reset_index(drop=True),
        'touch_quality_score': first_touch_df['touch_quality_score'].reset_index(drop=True),  # 항목0

        'setter': set_df['player_id'].reset_index(drop=True),
        'settercall': set_df['set_code'].reset_index(drop=True),
        'settercall_desc': set_df['settercall_desc'].reset_index(drop=True),   # 항목16
        'settercall_type': set_df['settercall_type'].reset_index(drop=True),   # 항목16

        'score_diff': score_diff_arr,      # 항목1
        'score_stage': game_stages,        # 항목2
        'is_clutch': is_clutch,            # 항목13

        'prev_attack_result': prev_attack_results.reset_index(drop=True),  # 항목9
        'streak_flow': set_df['actual_streak_flow'].reset_index(drop=True),
        'possession_num': set_df['possession_number'].reset_index(drop=True),  # 항목7

        'point_type': point_type,          # 항목12
        'own_rotation': own_rot,           # 항목5
        'opp_rotation': opp_rot,           # 항목14
        'setter_front': [1 if int(r) in [2, 3, 4] else 0 for r in own_rot],  # 항목8

        'attacker_rt_eff': attack_df['rt_atk_eff'].reset_index(drop=True),  # 항목3
        'is_repeat_route': is_repeat_route,                                  # 항목4
        'consecutive_toss_count': consecutive_toss_count,                    # 항목11

        'attack_code': attack_df['attack_code'].reset_index(drop=True),
        'attack_zone': attack_df['attack_zone'].reset_index(drop=True),
        'num_blockers': attack_df['num_blockers'].reset_index(drop=True),
        'attack_between_blockers': attack_df['attack_between_blockers'].reset_index(drop=True),
        'attacker_name': attack_df['player_id'].reset_index(drop=True),
        'attack_result': attack_df['evaluation_code'].reset_index(drop=True),

        'attacker_season_pct': attack_df['season_attack_pct'].reset_index(drop=True),   # 항목15
        'ace_is_front': set_df['ace_is_front'].reset_index(drop=True),                   # 항목10: 세트하는 팀의 에이스가 지금 전위인지
    })

    return final_data


# ================================================================
# 4. 배치 실행 (126개 파일 순회)
# ================================================================

def run_batch(folder_path, target_setters, pattern='*.dvw', encoding='cp949',
              season_stats_csv=None, verbose=True):
    """
    folder_path 안의 모든 .dvw 파일을 순회해서 처리하고 하나로 합침.
    반환값: (합쳐진 DataFrame, 에러/스킵 로그 리스트)

    target_setters: {'TEAM_C': [2, 3], 'TEAM_G': [2], ...} 형태 권장.
      해당 팀+번호 선수가 특정 경기 로스터에 없으면(예: 부상/결장) 그 경기는 자동 스킵되고
      에러가 아니라 스킵 로그에 남습니다 — 전체 배치는 계속 진행됩니다.
    season_stats_csv: 시즌 공격통계 CSV 경로. 주면 10번(에이스 전/후위), 15번(시즌 성공률) 컬럼이 채워짐.
    """
    season_df, ace_df = (None, None)
    if season_stats_csv is not None:
        season_df, ace_df = load_season_stats(season_stats_csv)

    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if verbose:
        print(f"총 {len(files)}개 파일 발견")

    all_results = []
    error_log = []

    for i, fp in enumerate(files, 1):
        try:
            df = process_single_file(fp, target_setters, encoding=encoding,
                                      season_df=season_df, ace_df=ace_df)
            if df.empty:
                error_log.append({'file': fp, 'error': '대상 세터의 유효 시퀀스 없음 (스킵)'})
                if verbose:
                    print(f"[{i}/{len(files)}] {os.path.basename(fp)} -> 시퀀스 0개, 스킵")
                continue
            all_results.append(df)
            if verbose:
                print(f"[{i}/{len(files)}] {os.path.basename(fp)} -> {len(df)}행")
        except Exception as e:
            error_log.append({'file': fp, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()})
            if verbose:
                print(f"[{i}/{len(files)}] {os.path.basename(fp)} -> 실패: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    if verbose:
        print(f"\n완료: 성공 {len(all_results)}개 파일, 실패/스킵 {len(error_log)}개 파일, 총 {len(final_df)}행")
        if error_log:
            print("\n실패/스킵 목록:")
            for e in error_log:
                print(f"  - {os.path.basename(e['file'])}: {e['error']}")

    return final_df, error_log


if __name__ == '__main__':
    # 사용 예시 (플레이스홀더 팀명/등번호 - 실제 실행 시 본인 데이터에 맞게 교체하세요)
    TARGET_SETTERS = {
        'TEAM_A': [2],
        'TEAM_D': [3, 20],
        'TEAM_G': [2],
        'TEAM_B': [2],
        'TEAM_C': [6],
        'TEAM_E': [6],
        'TEAM_F': [2, 3],
    }
    final_df, errors = run_batch(
        folder_path='/mnt/user-data/uploads',
        target_setters=TARGET_SETTERS,
        pattern='*.dvw',
        season_stats_csv='/mnt/user-data/outputs/season_attack_stats.csv',
    )
    print(final_df.shape)
    if not final_df.empty:
        out_path = '/mnt/user-data/outputs/final_dataset.csv'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {out_path}")
