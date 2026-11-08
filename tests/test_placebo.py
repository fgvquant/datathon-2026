"""Testes do placebo com dados simulados em que a resposta é conhecida."""
import io
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placebo import escolhe_horarios, le_klines, monta_janela  # noqa: E402

H6 = timedelta(hours=6)


def test_placebo_pula_semana_que_cruza_outro_ataque():
    t = datetime(2024, 3, 20, 14, 3, 27)
    ataques = {"A": t, "B": t - timedelta(weeks=1, hours=2), "C": datetime(2022, 1, 1)}
    escolha = escolhe_horarios(ataques, [1, 2], H6)
    assert escolha["A"] == (2, t - timedelta(weeks=2))  # 1 semana antes cai a 2h de B
    assert escolha["C"][0] == 1


def test_placebo_some_quando_nenhuma_semana_serve():
    t = datetime(2024, 3, 20)
    ataques = {"A": t, "B": t - timedelta(weeks=1), "C": t - timedelta(weeks=2)}
    assert "A" not in escolhe_horarios(ataques, [1, 2], H6)


def _zip(linhas):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.csv", "\n".join(",".join(map(str, l)) for l in linhas))
    return b.getvalue()


def test_le_klines_ms_e_us():
    t = datetime(2026, 1, 5, 0, 1)
    ms = int(t.replace(tzinfo=timezone.utc).timestamp() * 1000)
    base = [1, 2, 0.5, 1.5, 10, 0, 20, 3, 4, 5, 0]
    for ts in (ms, ms * 1000):
        assert le_klines(_zip([[ts, *base]]))[t] == ["1", "2", "0.5", "1.5", "10", "20", "3", "4", "5"]


def test_janela_tem_361_velas_e_preenche_ausentes():
    ref = datetime(2024, 3, 20, 14, 3, 27)
    m0 = datetime(2024, 3, 20, 14, 3)
    velas = {m0 + timedelta(minutes=t): ["1", "1", "1", str(100 + t), "5", "5", "1", "1", "1"]
             for t in range(-180, 181) if t not in (-180, 7)}
    j = monta_janela(velas, ref)
    assert len(j) == 361 and j[180][:2] == (0, m0)
    assert j[0][2][3] == "-79" and j[0][3]           # −180 ausente: usa o primeiro close (vela −179)
    assert j[187][2][:4] == ["106"] * 4 and j[187][3]  # +7 ausente: usa o close da +6
    assert j[187][2][4] == "0"
